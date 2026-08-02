/***************************************************************************************************
 * Stead State Two-Dimensional Parallel Direct Navier-Stokes Least Squares Solver
 * 
 * Authors: Boatwright, Taylor
 *          Breeden, Ja'Nya
 *          Lytch, Jada
 *          Brauss, Daniel
 *          
 * 
 * Solves the Stationary Navier-Stokes equations with Trilinos Direct Solver. 
 *
 * Contains an Exact Solution for Convergence Check.
 * 
 **************************************************************************************************/


#include <deal.II/base/quadrature_lib.h>        // needed for QGauss<dim> quadrature_formula and quadrature_formula_BiotSavart
#include <deal.II/base/timer.h>                 // needed for component_timers to time code parts
#include <deal.II/base/conditional_ostream.h>   // needed for pcout - stream output from processor (master) zero only    
#include <deal.II/base/convergence_table.h>

#include <deal.II/grid/grid_generator.h>        // needed for grid generation
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/distributed/tria.h>           // needed for parallel::distributed::Triangulation<dim> triangulation

#include <deal.II/dofs/dof_handler.h>           // needed for DoFHandler<dim> dof_handler 
#include <deal.II/dofs/dof_tools.h>             // needed for DoFTools::count_dofs_per_block in MHDProblem<dim>::setup_partitioning()
#include <deal.II/dofs/dof_renumbering.h>       // needed for DoFRenumbering::Hierarchical and DoFRenumbering::component_wise

#include <deal.II/fe/fe_q.h>                    // needed for FE_Q<dim> in constructor initialization of fe
#include <deal.II/fe/fe_values.h>               // needed for FEValues<dim> stokes_ohms_fe_values in
                                                // MHDProblem<dim>::assemble_block_stokes_ohms_mass_matrix
#include <deal.II/fe/fe_system.h>               // needed for FESystem<dim> fe 
#include <deal.II/fe/fe_nedelec.h>              // needed for FE_Nedelec<dim> in constructor initialization of fe

#include <deal.II/lac/solver_gmres.h>           // needed for MHDProblem<dim>::solve_block_system_iteratively()
#include <deal.II/lac/affine_constraints.h>      // needed for ConstratinMatrix constraints
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/numerics/vector_tools.h>      // needed for VectorTools::interpolate_boundary_values 
                                                // in MHDProblem<dim>::setup_constraints()
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>          // needed for MHDProblem::output_results_parallel(unsigned int) 
                                                // DataComponentInterpretation

#include <iostream>

#include <Epetra_FECrsGraph.h>
#include <Epetra_Map.h>
#include <Epetra_MpiComm.h>
#include "mpi.h"

namespace LeastSquaresNavierStokes
{
  using namespace dealii;

/*******************************************************************************
 *  * Define boundary conditions, initial conditions, forcing function and 
 *   * exact solution for convergence study
 *    ******************************************************************************/
template<int dim>
class ExactSolutionBoundaryValues : public Function<dim>
{
    public:
        ExactSolutionBoundaryValues() : Function<dim>(3) {}
        virtual void vector_value(const Point<dim> &p,
                Vector<double> &values) const;
};

template<int dim>
void ExactSolutionBoundaryValues<dim>::vector_value(const Point<dim> &p,
                Vector<double> &values) const
{
  double x = p[0];
  double y = p[1];

  values(0) = exp(x)*sin(y);
  values(1) = exp(x)*cos(y);
  //  values(2) = -0.5*exp(2*x) + 0.5; 
  values(2) = -0.5*exp(2*x) + 0.5 + exp(-1.*y*y) - 1.;
}

template<int dim>
class ExactSolutionForcingFunction : public Function<dim>
{
    public:
        ExactSolutionForcingFunction() : Function<dim>(3) {};
        virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const;
};

template<int dim>
void ExactSolutionForcingFunction<dim>::vector_value(const Point<dim> &p,
                Vector<double> &values) const
{

  //const double nu = 1;
  //const double x = p[0];
  //const double y = p[1];
  
  values(0) = 0.;
  //  values(1) = 0.;
  values(1) = -p[1]*exp(-1.*p[1]*p[1]);
  values(2) = 0.;
  //  values(2) = exp(p[1]);
  
}

template<int dim>
class ExactSolution : public Function<dim>
{
    public:
        ExactSolution() : Function<dim>(3) {}
        virtual void vector_value(const Point<dim> &p,
                Vector<double> &values) const;
        virtual void vector_gradient(const Point<dim> &p,
                std::vector<Tensor<1,dim> > &gradients) const;
};

template<int dim>
void ExactSolution<dim>::vector_value(const Point<dim> &p,
                Vector<double> &values) const
{

  double x = p[0];
  double y = p[1];

  values(0) = exp(x)*sin(y);
  values(1) = exp(x)*cos(y);
  //  values(2) = -0.5*exp(2*x) + 0.5; 
  values(2) = -0.5*exp(2*x) + 0.5 + exp(-1.*y*y) - 1.;
  //  values(2) = -0.5*exp(2*x) + 0.5 + exp(y) - 1; 

}

template<int dim>
void ExactSolution<dim>::vector_gradient(const Point<dim> &p,
                std::vector<Tensor<1,dim> > & gradients) const
{

  double x = p[0];
  double y = p[1];

  gradients[0][0] = exp(x)*sin(y);  // u_1 gradient
  gradients[0][1] = exp(x)*cos(y);

  gradients[1][0] = exp(x)*cos(y);  // u_2 gradient
  gradients[1][1] = -exp(x)*sin(y);

  gradients[2][0] = -exp(2*x);      // p gradient
  //  gradients[2][1] = 0.0;
  gradients[2][1] = -y*exp(-1.*y*y);

}



  /*******************************************************************************
   *  * Define boundary conditions, initial conditions, forcing function and 
   *  * for Lid Driven Cavity Problem
   *  ******************************************************************************/
  template<int dim>
  class LidDrivenCavityBoundaryValues : public Function<dim>
  {
    public:
        LidDrivenCavityBoundaryValues() : Function<dim>(3) {}
        virtual void vector_value(const Point<dim> &p,
                Vector<double> &values) const override;
  };

  template<int dim>
  void LidDrivenCavityBoundaryValues<dim>::vector_value(const Point<dim> &p,
                Vector<double> &values) const
  {

    double x = p[0];
    double y = p[1];
    const double pi = 3.141592654;
    double a = 1./32.;

    if (y == 1.0)  // at moving wall/lid of unit square boundary
    {  // note: domain is [0,1]^2 of xy-plane
      if (fabs(x-1.0) > a && fabs(x-0.0) > a)  // a < x < 1-a since x in [0,1]^2
      {
        values(0) = 1.0;
	values(1) = 0.0;
      }
      else if (fabs(x-0.) <= a)  // 0 <= x <= a
      {
        values(0) = std::sin(x*pi/(2*a));
	values(1) = 0.0;
      }
      else if (fabs(x-1.0) <= a) // 1-a <= x <= 1
      {
        values(0) = std::sin((1-x)*pi/(2*a));
	values(1) = 0.0;
      }
    }
    else // y != 1.0 :  0 < y < 1 and 0 <= x <= 1
    {
      values(0) = 0.0;
      values(1) = 0.0;
    }

  }

  template<int dim>
  class LidDrivenCavityForcingFunction : public Function<dim>
  {
    public:
        LidDrivenCavityForcingFunction() : Function<dim>(3) {};
        virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override;
  };

  template<int dim>
  void LidDrivenCavityForcingFunction<dim>::vector_value(const Point<dim> &p,
                Vector<double> &values) const
  {

    values(0) = 0.;
    values(1) = 0.;
    values(2) = 0.;
  
  }


  // This is a place holder function called in calculate_errors member function
  // since is not an exact solution for the Lid-Driven Cavity problem
  template<int dim>
  class LidDrivenCavityExactSolution : public Function<dim>
  {
    public:
      LidDrivenCavityExactSolution() : Function<dim>(3) {}
      virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;
      virtual void vector_gradient(const Point<dim> &p,
                                   std::vector<Tensor<1,dim> > &gradients) const override;
  };

  template<int dim>
  void LidDrivenCavityExactSolution<dim>::vector_value(const Point<dim> &p,
                  Vector<double> &values) const
  {

    values(0) = 0.;
    values(1) = 0.;
    values(2) = 0.;

  }

  template<int dim>
  void LidDrivenCavityExactSolution<dim>::vector_gradient(const Point<dim> &p,
                  std::vector<Tensor<1,dim> > & gradients) const
  {

    gradients[0][0] = 0.0;
    gradients[0][1] = 0.0;

    gradients[1][0] = 0.0;
    gradients[1][1] = 0.0;

    gradients[2][0] = 0.0;
    gradients[2][1] = 0.0;

  }



  // Glowinski Navier-Stokes Least Squares Solver 
  template <int dim>
  class NavierStokesSolver
  {
    public:

      NavierStokesSolver();
      void run_glowinski_algorithm();

    private:
      void glowinski_step(const TrilinosWrappers::MPI::Vector &src,
		          TrilinosWrappers::MPI::Vector &dst);
      void glowinski_initial_solve_yn_gn_wn();
      void glowinski_initial_guess();
      void setup_glowinski_systems();

      void assemble_stokes_y_n_system();
      void assemble_stokes_system(std::string system_soln_name);
      void post_process_pressure();
      void output_data();

      void setup_mesh_dofs_and_systems();          // Step 1 
      void assemble_stokes_matrix_and_rhs();       // Step 2
      void solve_stokes_initial_guess();           // Step 3
     
      void solve_stokes_system(std::string system_soln_name);
 
      double calculate_bilinear_a_realnumber(std::string system_soln_name);
      void assemble_trilinear_c_rhs_vector(std::string first_term,
                                           std::string second_term,
                                           std::string third_term);
      Tensor<1,3> calculate_derivs_of_j_n_at_rho(double rho);

      void solve_stokes_system_y_n();

      void calculate_errors(std::string solution_of_interest);
      void print_errors(); 

      // variables communication
      ConditionalOStream    pcout;
      MPI_Comm              mpi_communicator;
      const unsigned int    n_mpi_processes;
      const unsigned int    this_mpi_process;

      // variables fem
      parallel::distributed::Triangulation<dim>    triangulation;
      DoFHandler<dim>                              dof_handler;
      FESystem<dim>                                fe;

      AffineConstraints<double>    constraints_zero_bdry_conditions;
      AffineConstraints<double>    constraints_liddriven_bdry_conditions;

      ConvergenceTable             velocity_convergence_table;
      ConvergenceTable             pressure_convergence_table;

      TrilinosWrappers::SparseMatrix  stokes_matrix_zero_boundary;
      TrilinosWrappers::SparseMatrix  stokes_matrix_liddriven_boundary;

      TrilinosWrappers::SparseMatrix  laplace_velocity_mass_matrix;

      TrilinosWrappers::SparseMatrix  stokes_system_matrix;

      TrilinosWrappers::MPI::Vector   system_rhs_zero_boundary;
      TrilinosWrappers::MPI::Vector   system_rhs_liddriven_boundary;
      TrilinosWrappers::MPI::Vector   stokes_system_rhs;
      TrilinosWrappers::MPI::Vector   trilinear_c_rhs_vector;

      TrilinosWrappers::MPI::Vector   system_solution_u_n;
      TrilinosWrappers::MPI::Vector   system_solution_y_n;
      TrilinosWrappers::MPI::Vector   system_solution_g_n;
      TrilinosWrappers::MPI::Vector   system_solution_w_n;
      TrilinosWrappers::MPI::Vector   system_solution_y_1_n;
      TrilinosWrappers::MPI::Vector   system_solution_y_2_n;

      double a_of_y_0_y_0_value;
      double a_of_y_0_y_0_value_on_process;
      double a_of_y_n_y_n_value;
      double a_of_g_0_g_0_value;
      double a_of_g_0_g_0_value_on_process;
      double a_of_g_n_g_n_value;
      double a_of_g_n_g_n_value_on_process;
      double a_of_g_n_g_n_value_old;

      double J_n_rel_error;

      // This vector is needed for vector multiply with velocity_laplace_mass_matrix
      // This solution vector's size is different than the other solution vectors.
      // Its vector size is just number of locally owned dofs for all processes
      // as opposed to number of locally relevant dofs for all processes
      // like the right-hand side vectors and the other solution vectors (the 
      // other solution vectors are used in assembling the system - in fe_values 
      // calls to functions like get_function_values and get_function_gradients) 
      TrilinosWrappers::MPI::Vector   system_solution_index_set;

      
      std::vector<unsigned int>    system_matrix_block_sizes;
      std::vector<IndexSet>        ns_partitioning;
      std::vector<IndexSet>        ns_relevant_partitioning;
      IndexSet                     ns_index_set;
      IndexSet                     ns_relevant_set;

      bool          rebuild_block_stokes_matrix;

      double        nu;
      unsigned int  subdivisions_per_coord_direction;

  };

  /* Constructor */
  template <int dim>
  NavierStokesSolver<dim>::NavierStokesSolver()
      :
      pcout (std::cout,
             (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)),
      mpi_communicator (MPI_COMM_WORLD),
      n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
      triangulation (MPI_COMM_WORLD,
                     typename Triangulation<dim>::MeshSmoothing
                     (Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening)),
      dof_handler (triangulation),
      fe(FE_Q<dim>(2), dim,  // velocity
         FE_Q<dim>(1), 1),    // pressure
      J_n_rel_error(1.),
      rebuild_block_stokes_matrix(true),
      nu(1.000),
      subdivisions_per_coord_direction(32)
  {}


  template <int dim>
  void NavierStokesSolver<dim>::output_data ()
  {

    pcout << "Outputting data..." << std::endl;

    DataOut<dim>  data_out;
    data_out.attach_dof_handler (dof_handler);

    std::vector<std::string> solution_names (dim, "velocity");
    solution_names.push_back ("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation
      (dim, DataComponentInterpretation::component_is_part_of_vector);  // velocity vector

    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);  // pressure

    data_out.add_data_vector (system_solution_u_n, 
                              solution_names, 
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);

    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches ();

    data_out.write_vtu_with_pvtu_record("./",  // directory writing to
                                        "solution-Re200-64x64-15May2025", // filename
                                        1, // timestep counter index 
					mpi_communicator, // MPI comm
                                        2, // digits to use for counter
					2); // number files created
  }


  template <int dim>
  void NavierStokesSolver<dim>::post_process_pressure ()
  {
    pcout << "Post processing the pressure..." << std::endl;

    const double pressure_mean_value = VectorTools::compute_mean_value (dof_handler,
                                                                        QGauss<dim>(3),
                                                                        system_solution_u_n,
                                                                        dim);

    {
      pcout << "pressure_mean_value = " << pressure_mean_value << std::endl;
      TrilinosScalar  pressure_shift = -1.0*pressure_mean_value;

      TrilinosWrappers::MPI::Vector system_solution_reuse_vec1 (ns_index_set,
                                                                  MPI_COMM_WORLD);
      system_solution_reuse_vec1 = system_solution_u_n;
      
      // adding the pressure_shift to all the
      // pressures in the pressure block of
      // the distributed solution
      const FEValuesExtractors::Scalar pressure(dim);
      IndexSet local_pressure_dofs = DoFTools::extract_dofs(dof_handler,
                                                            fe.component_mask(pressure));
      for (auto index = local_pressure_dofs.begin(); 
              index != local_pressure_dofs.end(); ++index)
        system_solution_reuse_vec1(*index) += pressure_shift;
      
      system_solution_u_n = system_solution_reuse_vec1;     

    }
                                
  }


  //  Simplifying
  //
  //    j_n(rho) = 1/2 a(y^n(rho), y^n(rho)) 
  //             = 1/2 a(y^n - rho y_1^n + rho^2 y_2^n,y^n - rho y_1^n + rho^2 y_2^n)
  //                         1                2                    3
  //             = 1/2 [ a(y^n, y^n) - rho a(y^n, y_1^n) + rho^2 a(y^n, y_2^n)
  //                                  4                   5                       6
  //                      - rho ( a(y_1^n, y^n) - rho a(y_1^n, y_1^n) + rho^2 a(y_1^n, y_2^n)
  //                                    7                   8                         9
  //                      + rho^2 ( a(y_2^n, y^n) - rho a(y_2^n, y_1^n) + rho^2  a(y_2^n, y_2^n) ]
  //                          1                  2/4                    3/7
  //             = 1/2 [ a(y^n, y^n) - 2 rho a(y^n, y_1^n) + 2 rho^2 a(y^n, y_2^n)
  //                                  5                        6/8                     9
  //                      + rho^2 a(y_1^n, y_1^n) - 2 rho^3 a(y_1^n, y_2^n) + rho^4 a(y_2^n, y_2^n) ]
  //
  //  Therefore
  //
  //    j_n'(rho) = 1/2 d/d rho [ a(y^n, y^n) - 2 rho a(y^n, y_1^n) + 2 rho^2 a(y^n, y_2^n)
  //                               + rho^2 a(y_1^n, y_1^n) - 2 rho^3 a(y_1^n, y_2^n) 
  //                               + rho^4 a(y_2^n, y_2^n) ]
  //              = 1/2 [ -2 a(y^n, y_1^n) + 4 rho a(y^n, y_2^n) + 2 rho a(y_1^n, y_1^n)
  //                         - 6 rho^2 a(y_1^n, y_2^n) + 4 rho^3 a(y_2^n, y_2^n) ]
  //
  //  And
  //
  //    j_n''(rho) = 1/2 d/d rho [ -2 a(y^n, y_1^n) + 4 rho a(y^n, y_2^n) + 2 rho a(y_1^n, y_1^n)
  //                                    - 6 rho^2 a(y_1^n, y_2^n) + 4 rho^3 a(y_2^n, y_2^n) ]
  //               = 1/2 [ 4 a(y^n, y_2^n) + 2 a(y_1^n, y_1^n) 
  //                            - 12 rho a(y_1^n, y_2^n)  + 12 rho^2 a(y_2^n, y_2^n) ]
  //
  template<int dim>
  Tensor<1,3> NavierStokesSolver<dim>::calculate_derivs_of_j_n_at_rho(double rho)
  {
    pcout << "          Calculating the Derivatives of j_n = j_n(rho) for Glowinski Method..."
          << std::endl;

    Tensor<1,3> derivs_j_n_at_rho;

    double zeroth_deriv_y_n_of_rho = 0;
    double first_deriv_y_n_of_rho = 0;
    double second_deriv_y_n_of_rho = 0;

    double a_of_y_n_y_n = 0;
    double a_of_y_n_y_n_process = 0;
    double a_of_y_n_y_1_n = 0;
    double a_of_y_n_y_1_n_process = 0;
    double a_of_y_n_y_2_n = 0;
    double a_of_y_n_y_2_n_process = 0;
    double a_of_y_1_n_y_1_n = 0;
    double a_of_y_1_n_y_1_n_process = 0;
    double a_of_y_1_n_y_2_n = 0;
    double a_of_y_1_n_y_2_n_process = 0;
    double a_of_y_2_n_y_2_n = 0;
    double a_of_y_2_n_y_2_n_process = 0;

    QGauss<dim>                 quadrature_formula(4);

    const int                   dofs_per_cell = fe.dofs_per_cell;
    const int                   n_q_points = quadrature_formula.size();

    std::vector<Tensor<2,dim> >        y_n_velocity_gradients(n_q_points);
    std::vector<Tensor<2,dim> >        y_1_n_velocity_gradients(n_q_points);
    std::vector<Tensor<2,dim> >        y_2_n_velocity_gradients(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients |
                                update_JxW_values | update_quadrature_points);

    typename DoFHandler<dim>::active_cell_iterator
                        cell = dof_handler.begin_active(), endc = dof_handler.end();

    for (; cell!=endc; ++cell)
    {
      if (cell->is_locally_owned())
      {

        fe_values.reinit(cell);

        fe_values[velocities].get_function_gradients(system_solution_y_n, y_n_velocity_gradients);
        fe_values[velocities].get_function_gradients(system_solution_y_1_n, y_1_n_velocity_gradients);
        fe_values[velocities].get_function_gradients(system_solution_y_2_n, y_2_n_velocity_gradients);

        //calculate cell contributions through quadrature
        for (int q = 0; q < n_q_points; q++)
        {
          a_of_y_n_y_n_process += (nu*scalar_product(y_n_velocity_gradients[q],
                                   y_n_velocity_gradients[q])) * fe_values.JxW(q);
          a_of_y_n_y_1_n_process += (nu*scalar_product(y_n_velocity_gradients[q],
                                     y_1_n_velocity_gradients[q])) * fe_values.JxW(q);
          a_of_y_n_y_2_n_process += (nu*scalar_product(y_n_velocity_gradients[q],
                                     y_2_n_velocity_gradients[q])) * fe_values.JxW(q);
          a_of_y_1_n_y_1_n_process += (nu*scalar_product(y_1_n_velocity_gradients[q],
                                       y_1_n_velocity_gradients[q])) * fe_values.JxW(q);
          a_of_y_1_n_y_2_n_process += (nu*scalar_product(y_1_n_velocity_gradients[q],
                                       y_2_n_velocity_gradients[q])) * fe_values.JxW(q);
          a_of_y_2_n_y_2_n_process += (nu*scalar_product(y_2_n_velocity_gradients[q],
                                       y_2_n_velocity_gradients[q])) * fe_values.JxW(q);
        } // quadrature loop
        
      } // if cell locally owned
        
    } // cell loop
        
    double rho_squared = rho * rho;
    double rho_cubed = rho_squared * rho;
    double rho_fourth = rho_cubed * rho;

    a_of_y_n_y_n = Utilities::MPI::sum(a_of_y_n_y_n_process, MPI_COMM_WORLD);
    a_of_y_n_y_1_n = Utilities::MPI::sum(a_of_y_n_y_1_n_process, MPI_COMM_WORLD);
    a_of_y_n_y_2_n = Utilities::MPI::sum(a_of_y_n_y_2_n_process, MPI_COMM_WORLD);
    a_of_y_1_n_y_1_n = Utilities::MPI::sum(a_of_y_1_n_y_1_n_process, MPI_COMM_WORLD);
    a_of_y_1_n_y_2_n = Utilities::MPI::sum(a_of_y_1_n_y_2_n_process, MPI_COMM_WORLD);
    a_of_y_2_n_y_2_n = Utilities::MPI::sum(a_of_y_2_n_y_2_n_process, MPI_COMM_WORLD);

    //j_n(rho) = 1/2 [ a(y^n, y^n) - 2 rho a(y^n, y_1^n) + 2 rho^2 a(y^n, y_2^n)
    //                      + rho^2 a(y_1^n, y_1^n) - 2 rho^3 a(y_1^n, y_2^n) + rho^4 a(y_2^n, y_2^n) ]
    zeroth_deriv_y_n_of_rho = 0.5 * (a_of_y_n_y_n - 2. * rho * a_of_y_n_y_1_n
                              + 2. * rho_squared * a_of_y_n_y_2_n + rho_squared * a_of_y_1_n_y_1_n
                              - 2. * rho_cubed * a_of_y_1_n_y_2_n + rho_fourth * a_of_y_2_n_y_2_n );
    //j_n'(rho) = 1/2 [ -2 a(y^n, y_1^n) + 4 rho a(y^n, y_2^n) + 2 rho a(y_1^n, y_1^n)
    //                         - 6 rho^2 a(y_1^n, y_2^n) + 4 rho^3 a(y_2^n, y_2^n) ]
    first_deriv_y_n_of_rho  = 0.5 * (-2. * a_of_y_n_y_1_n + 4. * rho * a_of_y_n_y_2_n
                              + 2. * rho * a_of_y_1_n_y_1_n - 6. * rho_squared * a_of_y_1_n_y_2_n
                              + 4. * rho_cubed * a_of_y_2_n_y_2_n );
    //j_n''(rho) = 1/2 [ 4 a(y^n, y_2^n) + 2 a(y_1^n, y_1^n) 
    //                            - 12 rho a(y_1^n, y_2^n)  + 12 rho^2 a(y_2^n, y_2^n) ]
    second_deriv_y_n_of_rho  = 0.5 * (4. * a_of_y_n_y_2_n + 2. * a_of_y_1_n_y_1_n
                               - 12. * rho * a_of_y_1_n_y_2_n + 12. * rho_squared * a_of_y_2_n_y_2_n );
    
    derivs_j_n_at_rho[0] = zeroth_deriv_y_n_of_rho;
    derivs_j_n_at_rho[1] = first_deriv_y_n_of_rho;
    derivs_j_n_at_rho[2] = second_deriv_y_n_of_rho;
    
    return derivs_j_n_at_rho;
    
  }
    

  // Setting up right-hand sides for Steps 3, 5, and 6
  // 
  // Step 3 - Solve for g^0 in V_0
  //  -> a(g^0, z) + b(z, theta) = a(y^0, z) + c(u^0, z, y^0) + c(z, u^0, y^0)
  //                b(g^0, q)   = 0 
  // Step 5 - Solve y_1^n in V_0
  //  -> a(y_1^n, z) + b(z, theta) = a(w^n, z) + c(u^n, w^n, z) + c(w^n, u^n, z)
  //                b(y_1^n, q)   = 0 
  // Step 6 - Solve y_2^n in V_0
  //  -> a(y_2^n, z) + b(z, theta) = c(w^n, w^n, z) 
  //                b(y_2^n, q)   = 0 
  template<int dim>
  void NavierStokesSolver<dim>::assemble_stokes_system(std::string system_soln_name)
  {
    pcout << "Assembling Stokes " << system_soln_name 
          << " System for Glowinski Method..." << std::endl;

    system_rhs_zero_boundary = 0;

    //LidDrivenCavityForcingFunction<dim>  forcing_function;              
    ExactSolutionForcingFunction<dim>  forcing_function;

    QGauss<dim>                   quadrature_formula(2);
    QGauss<dim-1>                 face_quadrature_formula(3);

    const int                     dofs_per_cell = fe.dofs_per_cell;
    const int                     n_q_points = quadrature_formula.size();

    std::vector<Vector<double> >  rhs_values (n_q_points, Vector<double>(dim+1));

    // solution values
    std::vector<Tensor<1,dim> >           solution_u_n_velocity_values(n_q_points);
    std::vector<Tensor< 2, dim> >         solution_u_n_velocity_gradients(n_q_points);
    std::vector<Tensor<1,dim> >           solution_y_n_velocity_values(n_q_points);
    std::vector<Tensor< 2, dim> >         solution_y_n_velocity_gradients(n_q_points);
    std::vector<Tensor<1,dim> >           solution_w_n_velocity_values(n_q_points);
    std::vector<Tensor< 2, dim> >         solution_w_n_velocity_gradients(n_q_points);
  
    std::vector<Tensor<2,dim> >   grad_phi_u (dofs_per_cell);
    std::vector<double>           div_phi_u (dofs_per_cell);
    std::vector<double>           phi_p (dofs_per_cell);
    std::vector<Tensor<1,dim> >   phi_u (dofs_per_cell);
  
    Vector<double>                cell_rhs(dofs_per_cell);

    FullMatrix<double>            cell_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>            cell_mass (dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients |
                                update_JxW_values | update_quadrature_points);

    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, update_values
                                        | update_quadrature_points | update_gradients |
                                        update_JxW_values);

    typename DoFHandler<dim>::active_cell_iterator
                        cell = dof_handler.begin_active(), endc = dof_handler.end();

    for (; cell!=endc; ++cell)
    {
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_mass   = 0;
        cell_rhs = 0;

        if (system_soln_name == "g_n")
        {
          fe_values[velocities].get_function_values(system_solution_u_n, solution_u_n_velocity_values);
          fe_values[velocities].get_function_gradients(system_solution_u_n, solution_u_n_velocity_gradients);
          fe_values[velocities].get_function_values(system_solution_y_n, solution_y_n_velocity_values);
          fe_values[velocities].get_function_gradients(system_solution_y_n, solution_y_n_velocity_gradients);
        }
        else if (system_soln_name == "y_1_n")
        {
          fe_values[velocities].get_function_values(system_solution_u_n, solution_u_n_velocity_values);
          fe_values[velocities].get_function_gradients(system_solution_u_n, solution_u_n_velocity_gradients);
          fe_values[velocities].get_function_values(system_solution_w_n, solution_w_n_velocity_values);
          fe_values[velocities].get_function_gradients(system_solution_w_n, solution_w_n_velocity_gradients);
        }
        else if (system_soln_name == "y_2_n")
        {
          fe_values[velocities].get_function_values(system_solution_w_n, solution_w_n_velocity_values);
          fe_values[velocities].get_function_gradients(system_solution_w_n, solution_w_n_velocity_gradients);
        }
        else
        {
          pcout << "Exception : Case Not Implemented for assemble_stokes_system argument..."
                << std::endl;
        }
  
        forcing_function.vector_value_list(fe_values.get_quadrature_points(), rhs_values);

        for (int q = 0; q < n_q_points; q++)
        {
          for (int k=0; k<dofs_per_cell; k++)
          {
            grad_phi_u[k] = fe_values[velocities].gradient (k, q);
            div_phi_u[k]  = fe_values[velocities].divergence (k, q);
            phi_p[k]      = fe_values[pressure].value (k, q);
            phi_u[k]      = fe_values[velocities].value (k, q);
          }

          for (int i = 0; i < dofs_per_cell; i++)
          {
            for (int j = 0; j < dofs_per_cell; j++)
            {
              cell_matrix(i,j) +=
                      (nu*scalar_product(grad_phi_u[i],grad_phi_u[j])
                      - phi_p[i]*div_phi_u[j]
                      - phi_p[j]*div_phi_u[i])
                                                *fe_values.JxW(q);

            }  // loop over columns

            //int equation_i = fe.system_to_component_index(i).first;

            if (system_soln_name == "g_n")
            {
              cell_rhs[i] +=
                            ( nu*scalar_product(grad_phi_u[i],solution_y_n_velocity_gradients[q])
                              + solution_u_n_velocity_values[q] * 
                                    transpose(grad_phi_u[i])*solution_y_n_velocity_values[q]
                              + phi_u[i]*
                                    transpose(solution_u_n_velocity_gradients[q])*
                                    solution_y_n_velocity_values[q]
                            )
                                        *fe_values.JxW(q);
            }
            else if (system_soln_name == "y_1_n")
            {
              cell_rhs[i] +=
                            ( nu*scalar_product(grad_phi_u[i],solution_w_n_velocity_gradients[q])
                              + solution_u_n_velocity_values[q] * 
                                    transpose(solution_w_n_velocity_gradients[q])*phi_u[i]
                              + solution_w_n_velocity_values[q] * 
                                    transpose(solution_u_n_velocity_gradients[q])*phi_u[i]
                            )
                                        *fe_values.JxW(q);
            }
            else if (system_soln_name == "y_2_n")
            {
              cell_rhs[i] +=
                            ( solution_w_n_velocity_values[q] * 
                                    transpose(solution_w_n_velocity_gradients[q])*phi_u[i]
                            )
                                        *fe_values.JxW(q);
            }
            else
            {
              pcout << "Exception : Case Not Implemented for assemble_stokes_system argument..."
                    << std::endl;
            }

          } // loop over rows

        }  // loop over quadrature points q

        cell->get_dof_indices(local_dof_indices);     // new
        constraints_zero_bdry_conditions.distribute_local_to_global (cell_rhs,
                                                                     local_dof_indices,
                                                                     system_rhs_zero_boundary,
                                                                     cell_matrix);
                                                                   
                                                                   
      } // end if over locally-owned cells
                                                                   
    } // loop over cells  
                                                                   
    system_rhs_zero_boundary.compress(VectorOperation::add);
 
  } 


  template<int dim>
  double NavierStokesSolver<dim>::calculate_bilinear_a_realnumber(std::string system_soln_name)
  {
    pcout << "         Calculating a(" << system_soln_name << "," << system_soln_name << ")"
          << " for relative error in Glowinski optimization loop..." << std::endl;
    Timer timer;
    timer.start ();

    double bilinear_a = 0;

    QGauss<dim>                 quadrature_formula(4);

    const int                   dofs_per_cell = fe.dofs_per_cell;
    const int                   n_q_points = quadrature_formula.size();

    std::vector<Tensor<2,dim> >        velocity_gradients_of_a(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients |
                                update_JxW_values | update_quadrature_points);

    typename DoFHandler<dim>::active_cell_iterator
                        cell = dof_handler.begin_active(), endc = dof_handler.end();

    for (; cell!=endc; ++cell)
    {
      if (cell->is_locally_owned())
      {

        fe_values.reinit(cell);
        if (system_soln_name == "y_n")
        {
          fe_values[velocities].get_function_gradients(system_solution_y_n, velocity_gradients_of_a);
        }
        else if (system_soln_name == "g_n")
        {
          fe_values[velocities].get_function_gradients(system_solution_g_n, velocity_gradients_of_a);
        }
        else
        {
          pcout << "Exception : Case Not Implemented for calculate_bilinear_a_realnumber argument..."
                << std::endl;
        }

        //calculate cell contributions through quadrature
        for (int q = 0; q < n_q_points; q++)
        {
          bilinear_a += (nu*scalar_product(velocity_gradients_of_a[q],
          velocity_gradients_of_a[q])) * fe_values.JxW(q);
        } // quadrature loop
        
      } // if cell locally owned
        
    } // cell loop
        
    return bilinear_a;
        
  }
       

  // Step 2 - Solve for y^n = y(u^n) in V_0
  //  -> - laplace y^n + grad sigma = - laplace u^n + u^n . grad u^n - f 
  //                        div y^n = 0
  template<int dim>
  void NavierStokesSolver<dim>::assemble_stokes_y_n_system()
  {
    pcout << "----------------------------------------------------" << std::endl;
    pcout << "Assembling Stokes y^n System for Glowinski Method..." << std::endl;
    pcout << "----------------------------------------------------" << std::endl;

    stokes_matrix_zero_boundary = 0;
    system_rhs_zero_boundary = 0;

    //LidDrivenCavityForcingFunction<dim>  forcing_function;  
    ExactSolutionForcingFunction<dim>  forcing_function;

    QGauss<dim>                   quadrature_formula(2);
    QGauss<dim-1>                 face_quadrature_formula(3);

    const int                     dofs_per_cell = fe.dofs_per_cell;
    const int                     n_q_points = quadrature_formula.size();

    std::vector<Vector<double> >  rhs_values (n_q_points, Vector<double>(dim+1));

    // solution values
    std::vector<Tensor<1,dim> >           solution_velocity_values(n_q_points);
    std::vector<Tensor< 2, dim> >         solution_velocity_gradients(n_q_points);

    std::vector<Tensor<2,dim> >   grad_phi_u (dofs_per_cell);
    std::vector<double>           div_phi_u (dofs_per_cell);
    std::vector<double>           phi_p (dofs_per_cell);
    std::vector<Tensor<1,dim> >   phi_u (dofs_per_cell);

    Vector<double>                cell_rhs(dofs_per_cell);

    FullMatrix<double>          cell_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>          cell_mass (dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients |
                                update_JxW_values | update_quadrature_points);

    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, update_values
                                        | update_quadrature_points | update_gradients |
                                        update_JxW_values);

    typename DoFHandler<dim>::active_cell_iterator
                        cell = dof_handler.begin_active(), endc = dof_handler.end();

    for (; cell!=endc; ++cell)
    {
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_mass   = 0;
        cell_rhs = 0;

        fe_values[velocities].get_function_values(system_solution_u_n, solution_velocity_values);
        fe_values[velocities].get_function_gradients(system_solution_u_n, solution_velocity_gradients);

        forcing_function.vector_value_list(fe_values.get_quadrature_points(), rhs_values);

        //calculate cell contribution to system         
        for (int q = 0; q < n_q_points; q++)
        {
          for (int k=0; k<dofs_per_cell; k++)
          {
            grad_phi_u[k] = fe_values[velocities].gradient (k, q);
            div_phi_u[k]  = fe_values[velocities].divergence (k, q);
            phi_p[k]      = fe_values[pressure].value (k, q);
            phi_u[k]      = fe_values[velocities].value (k, q);
          }

          for (int i = 0; i < dofs_per_cell; i++)
          {
            for (int j = 0; j < dofs_per_cell; j++)
            {
              cell_matrix(i,j) +=
                      (nu*scalar_product(grad_phi_u[i],grad_phi_u[j])
                      - phi_p[i]*div_phi_u[j]
                      - phi_p[j]*div_phi_u[i])
                                                *fe_values.JxW(q);
            }  // loop over columns

            int equation_i = fe.system_to_component_index(i).first;

            cell_rhs[i] +=
                          ( nu*scalar_product(grad_phi_u[i],solution_velocity_gradients[q])
                            + solution_velocity_values[q]*
                                  transpose(solution_velocity_gradients[q])*phi_u[i]
                            - fe_values.shape_value(i,q)*rhs_values[q](equation_i)
                          )
                                        *fe_values.JxW(q);

          } // loop over rows

        }  // loop over quadrature points q

        cell->get_dof_indices(local_dof_indices);     // new
        constraints_zero_bdry_conditions.distribute_local_to_global (cell_matrix,
                                                                     cell_rhs,
                                                                     local_dof_indices,
                                                                     stokes_matrix_zero_boundary,
                                                                     system_rhs_zero_boundary);
      } // end if over locally-owned cells

    } // loop over cells  

    stokes_matrix_zero_boundary.compress(VectorOperation::add);
    system_rhs_zero_boundary.compress(VectorOperation::add);

  }


  template<int dim>
  void NavierStokesSolver<dim>::assemble_stokes_matrix_and_rhs()
  {
    pcout << "//-------------------------------------------------" << std::endl; 
    pcout << "//Assembling Stokes System Matrix and RHS.........." << std::endl;
    pcout << "//-------------------------------------------------" << std::endl;

    //LidDrivenCavityForcingFunction<dim> forcing_function;
    ExactSolutionForcingFunction<dim> forcing_function;

    stokes_matrix_liddriven_boundary = 0;
    system_rhs_liddriven_boundary = 0;

    QGauss<dim>    quadrature_formula(2);
    QGauss<dim-1>  face_quadrature_formula(3);

    const unsigned int         dofs_per_cell = fe.dofs_per_cell;
    const unsigned int         n_q_points = quadrature_formula.size();

    std::vector<Vector<double> >  rhs_values(n_q_points, Vector<double>(dim+1));

    std::vector<Tensor<2,dim> >   grad_phi_u (dofs_per_cell);
    std::vector<double>           div_phi_u (dofs_per_cell);
    std::vector<double>           phi_p (dofs_per_cell);
    std::vector<Tensor<1,dim> >   phi_u (dofs_per_cell);

    FullMatrix<double>         cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>         cell_laplace_velocity_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>             cell_rhs(dofs_per_cell);
    std::vector<unsigned long>  local_dof_indices_of_cell(dofs_per_cell);
    
    FEValues<dim> fe_values(fe, quadrature_formula, update_quadrature_points |
                                update_values | update_gradients | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                         update_values | update_quadrature_points |
                                         update_gradients | update_normal_vectors |
                                         update_JxW_values);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices_of_cell);
        std::vector<Point<dim> > q_points = fe_values.get_quadrature_points();
        
        cell_matrix = 0;
        cell_laplace_velocity_mass_matrix = 0;
        cell_rhs = 0;

        forcing_function.vector_value_list(q_points, rhs_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            grad_phi_u[k] = fe_values[velocities].gradient(k,q);
            div_phi_u[k]  = fe_values[velocities].divergence(k,q);
            phi_p[k]      = fe_values[pressure].value(k,q);
            phi_u[k]      = fe_values[velocities].value(k,q);
          }

          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              cell_matrix(i,j) += (nu*scalar_product(grad_phi_u[i],grad_phi_u[j])
                                   - phi_p[i]*div_phi_u[j]
                                   - phi_p[j]*div_phi_u[i])*fe_values.JxW(q);

              cell_laplace_velocity_mass_matrix(i,j) 
                       += nu*scalar_product(grad_phi_u[i],grad_phi_u[j])*fe_values.JxW(q);
            } // loop over columns
   
            int equation_i = fe.system_to_component_index(i).first;

            cell_rhs[i] += fe_values.shape_value(i,q)*rhs_values[q](equation_i)
                            *fe_values.JxW(q);

          } // loop over rows

        } // quadrature point loop

        constraints_liddriven_bdry_conditions.distribute_local_to_global (cell_matrix,
                                                                          cell_rhs,
                                                                          local_dof_indices_of_cell,
                                                                          stokes_matrix_liddriven_boundary,
                                                                          system_rhs_liddriven_boundary);

      } // if locally owned cell 

    } // loop over cells

    stokes_matrix_liddriven_boundary.compress(VectorOperation::add);
    system_rhs_liddriven_boundary.compress(VectorOperation::add);

  }  


  template<int dim>
  void NavierStokesSolver<dim>::calculate_errors(std::string solution_of_interest)
  {
    pcout << "//--------------------------------" << std::endl;
    pcout << "//Computing Errors................" << std::endl;
    pcout << "//--------------------------------" << std::endl;

    //LidDrivenCavityExactSolution<dim>  exact_solution;
    ExactSolution<dim>  exact_solution;

    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0,dim),3);
    const ComponentSelectFunction<dim> pressure_mask(dim,3);

    Vector<float> difference_per_cell (triangulation.n_locally_owned_active_cells());

    QGauss<dim> quadrature(3);

    // velocity L2 cellwise errors
    if (solution_of_interest == "u_n")
      VectorTools::integrate_difference (dof_handler, system_solution_u_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &velocity_mask);
    else if (solution_of_interest == "y_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &velocity_mask);
    else if (solution_of_interest == "g_n")
      VectorTools::integrate_difference (dof_handler, system_solution_g_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &velocity_mask);
    else if (solution_of_interest == "y_1_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_1_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &velocity_mask);
    else if (solution_of_interest == "y_2_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_2_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &velocity_mask);
    else 
      pcout << "Exception: member function calculate_error argument Not Implemented..."
            << std::endl;

    unsigned int index = 0;
    double my_value = 0;

    {
      typename DoFHandler<dim>::active_cell_iterator
                       cell = dof_handler.begin_active(),
                       endc = dof_handler.end();
      for (; cell!=endc; ++cell, ++index)
      {
        if (cell->is_locally_owned())
        {
          my_value += difference_per_cell(index)*difference_per_cell(index);
        }
      }
    }

    const double u_l2_error = std::sqrt(Utilities::MPI::sum(my_value, MPI_COMM_WORLD));


    // velocity H1 cellwise errors
    if (solution_of_interest == "u_n")
      VectorTools::integrate_difference (dof_handler, system_solution_u_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &velocity_mask);
    else if (solution_of_interest == "y_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &velocity_mask);
    else if (solution_of_interest == "g_n")
      VectorTools::integrate_difference (dof_handler, system_solution_g_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &velocity_mask);
    else if (solution_of_interest == "y_1_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_1_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &velocity_mask);
    else if (solution_of_interest == "y_2_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_2_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &velocity_mask);
    else 
      pcout << "Exception: member function calculate_error argument Not Implemented..."
            << std::endl;

    index = 0;
    my_value = 0;

    {
      typename DoFHandler<dim>::active_cell_iterator
                       cell = dof_handler.begin_active(),
                       endc = dof_handler.end();
      for (; cell!=endc; ++cell, ++index)
      {
        if (cell->is_locally_owned())
        {
          my_value += difference_per_cell(index)*difference_per_cell(index);
        }
      }
    }

    const double u_H1_error = std::sqrt(Utilities::MPI::sum(my_value, MPI_COMM_WORLD));


    // pressure L2 cellwise errors
    if (solution_of_interest == "u_n")
      VectorTools::integrate_difference (dof_handler, system_solution_u_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &pressure_mask);
    else if (solution_of_interest == "y_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &pressure_mask);
    else if (solution_of_interest == "g_n")
      VectorTools::integrate_difference (dof_handler, system_solution_g_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &pressure_mask);
    else if (solution_of_interest == "y_1_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_1_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &pressure_mask);
    else if (solution_of_interest == "y_2_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_2_n, exact_solution, 
                                         difference_per_cell, quadrature,
                                         VectorTools::L2_norm, &pressure_mask);
    else 
      pcout << "Exception: member function calculate_error argument Not Implemented..."
            << std::endl;


    index = 0;
    my_value = 0;

    {
      typename DoFHandler<dim>::active_cell_iterator
                       cell = dof_handler.begin_active(),
                       endc = dof_handler.end();
      for (; cell!=endc; ++cell, ++index)
      {
        if (cell->is_locally_owned())
        {
          my_value += difference_per_cell(index)*difference_per_cell(index);
        }
      }
    }
    
    const double press_l2_error = 
           std::sqrt(Utilities::MPI::sum(my_value, MPI_COMM_WORLD));


    // pressure H1 cellwise errors
    if (solution_of_interest == "u_n")
      VectorTools::integrate_difference (dof_handler, system_solution_u_n, exact_solution,
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &pressure_mask);
    else if (solution_of_interest == "y_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_n, exact_solution,
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &pressure_mask);
    else if (solution_of_interest == "g_n")
      VectorTools::integrate_difference (dof_handler, system_solution_g_n, exact_solution,
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &pressure_mask);
    else if (solution_of_interest == "y_1_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_1_n, exact_solution,
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &pressure_mask);
    else if (solution_of_interest == "y_2_n")
      VectorTools::integrate_difference (dof_handler, system_solution_y_2_n, exact_solution,
                                         difference_per_cell, quadrature,
                                         VectorTools::H1_seminorm, &pressure_mask);
    else 
      pcout << "Exception: member function calculate_error argument Not Implemented..."
            << std::endl;

    index = 0;
    my_value = 0;

    {
      typename DoFHandler<dim>::active_cell_iterator
                       cell = dof_handler.begin_active(),
                       endc = dof_handler.end();
      for (; cell!=endc; ++cell, ++index)
      {
        if (cell->is_locally_owned())
        {
          my_value += difference_per_cell(index)*difference_per_cell(index);
        }
      }
    }

    const double press_H1_error = std::sqrt(Utilities::MPI::sum(my_value, MPI_COMM_WORLD));



    pcout << "u velocity L2 error for " << solution_of_interest 
          << " is : " << u_l2_error << std::endl;
    pcout << "u velocity H1 error for " << solution_of_interest 
          << " is : " << u_H1_error << std::endl;
    pcout << "p pressure L2 error for " << solution_of_interest
          << " is : " << press_l2_error << std::endl;
    pcout << "p pressure H1 error for " << solution_of_interest 
          << " is : " << press_H1_error << std::endl;

  }



  template<int dim>
  void NavierStokesSolver<dim>::solve_stokes_system(std::string system_soln_name)
  {
    pcout << "//----------------------------------------------------------" << std::endl;
    pcout << "//Solving Stokes System for " << system_soln_name << "......"
          <<  std::endl;
    pcout << "//----------------------------------------------------------" << std::endl;

    SolverControl solver_control (dof_handler.n_dofs(), 1e-13);
    TrilinosWrappers::SolverDirect solver_direct(solver_control);

    if (system_soln_name == "u_n")
    {
      TrilinosWrappers::MPI::Vector distributed_stokes_system_soln(system_rhs_liddriven_boundary);
      solver_direct.solve(stokes_system_matrix, 
                          distributed_stokes_system_soln, system_rhs_liddriven_boundary);
      constraints_liddriven_bdry_conditions.distribute (distributed_stokes_system_soln);
      system_solution_u_n = distributed_stokes_system_soln;
    }
    else if (system_soln_name == "y_n")
    {
      TrilinosWrappers::MPI::Vector distributed_stokes_system_soln(system_rhs_zero_boundary);
      solver_direct.solve(stokes_system_matrix, distributed_stokes_system_soln, 
                          stokes_system_rhs);
      constraints_zero_bdry_conditions.distribute (distributed_stokes_system_soln);
      system_solution_y_n = distributed_stokes_system_soln;
    }
    else if (system_soln_name == "g_n")
    {
      TrilinosWrappers::MPI::Vector distributed_stokes_system_soln(system_rhs_zero_boundary);
      solver_direct.solve(stokes_system_matrix, distributed_stokes_system_soln, stokes_system_rhs);
      constraints_zero_bdry_conditions.distribute (distributed_stokes_system_soln);
      system_solution_g_n = distributed_stokes_system_soln;
    }
    else if (system_soln_name == "y_1_n")
    {
      TrilinosWrappers::MPI::Vector distributed_stokes_system_soln(system_rhs_zero_boundary);
      solver_direct.solve(stokes_system_matrix, distributed_stokes_system_soln, stokes_system_rhs);
      constraints_zero_bdry_conditions.distribute (distributed_stokes_system_soln);
      system_solution_y_1_n = distributed_stokes_system_soln;
    }
    else if (system_soln_name == "y_2_n")
    {
      TrilinosWrappers::MPI::Vector distributed_stokes_system_soln(stokes_system_rhs);
      solver_direct.solve(stokes_system_matrix, distributed_stokes_system_soln, stokes_system_rhs);
      constraints_zero_bdry_conditions.distribute (distributed_stokes_system_soln);
      system_solution_y_2_n = distributed_stokes_system_soln;
    }
    else
    {
          pcout << "Exception : Case Not Implemented for solve_stokes_system argument..." << std::endl;
    }

  }


  template<int dim>
  void NavierStokesSolver<dim>::setup_glowinski_systems()
  {

    pcout << "//-----------------------------------------------" << std::endl;
    pcout << "Setting up Trinagulation and Distributing DOFs..." << std::endl;
    pcout << "//-----------------------------------------------" << std::endl;

    
    // bottom left corner of domain
    const Point<2> bottom_left = Point<2> (0.0,0.0);
    // top right corner of domain
    const Point<2> top_right = Point<2> (1.0,1.0);
    // subdividing domain
    std::vector<unsigned int> subdivisions(2,1);
    subdivisions[0] = subdivisions_per_coord_direction;
    subdivisions[1] = subdivisions_per_coord_direction;
    // generating grid
    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               subdivisions,
                                               bottom_left,
                                               top_right);


    dof_handler.distribute_dofs (fe);

    // NOTE: renumbering causes trouble with DataOut for visualization 
    //       (non-contiguous error vector addition error - since not block vectors)
    //DoFRenumbering::hierarchical(dof_handler);
    //DoFRenumbering::component_wise (dof_handler, system_matrix_sub_blocks);

    pcout << "         Number of degrees of freedom: " 
	  << dof_handler.n_dofs() << std::endl;


    pcout << "Step 0b. Viewing Matrix Partitioning..." << std::endl;


    
    pcout << "//-----------------------------------------------" << std::endl;
    pcout << "Setting up Matrix Block Partitioning............." << std::endl;
    pcout << "//-----------------------------------------------" << std::endl;

    // 2 velocity components and 1 pressure component - all components denoted 0
    std::vector<unsigned int>  system_matrix_sub_blocks(2 + 1, 0);
    // pressure component is 1 (and velocity components are 0 (above)) 
    system_matrix_sub_blocks[2] = 1;  // pressure component

    // count number of velocity and pressure dofs 
    const std::vector<unsigned long> system_matrix_dofs_per_block
	    = DoFTools::count_dofs_per_fe_block (dof_handler, 
                                                 system_matrix_sub_blocks);

    // setting up to place counts in matrix blocks (set block sizes)
    const unsigned int  n_u = system_matrix_dofs_per_block[0],
                        n_p = system_matrix_dofs_per_block[1];

    {

      ns_index_set = dof_handler.locally_owned_dofs();

      ns_partitioning.push_back(ns_index_set.get_view(0,n_u));
      ns_partitioning.push_back(ns_index_set.get_view(n_u,n_u+n_p));

      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              ns_relevant_set);

      ns_relevant_partitioning.push_back(ns_relevant_set.get_view(0,n_u));
      ns_relevant_partitioning.push_back(ns_relevant_set.get_view(n_u,n_u+n_p));

    }

    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "         Number of degrees of freedom: n_u + n_p = "
          << n_u + n_p << " with (n_u, n_p) = (" << n_u << ", " << n_p << ")"
          << std::endl;
    pcout.get_stream().imbue(s);

    pcout << "//-----------------------------------------------" << std::endl;
    pcout << "Setting Up Constraints..........................." << std::endl;
    pcout << "//-----------------------------------------------" << std::endl;

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    // the Glowinski algorithm requires zero boundary conditions
    // for the update (added to previous solution) so as not to 
    // change the boundary conditions set by the first solution
    constraints_zero_bdry_conditions.clear();
    // we can interchange between Exact Solution BCs and Lid Driven BCs
    constraints_liddriven_bdry_conditions.clear();

    constraints_zero_bdry_conditions.reinit(ns_relevant_set);
    constraints_liddriven_bdry_conditions.reinit(ns_relevant_set);

    VectorTools::interpolate_boundary_values(dof_handler,
                                                       0,
                                             //LidDrivenCavityBoundaryValues<dim>(),
                                             ExactSolutionBoundaryValues<dim>(),
                                             constraints_liddriven_bdry_conditions,
                                             fe.component_mask(velocities));

    VectorTools::interpolate_boundary_values(dof_handler,
                                                       0,
                                             Functions::ZeroFunction<dim>(3),
                                             constraints_zero_bdry_conditions,
                                             fe.component_mask(velocities));


    constraints_zero_bdry_conditions.close();
    constraints_liddriven_bdry_conditions.close();

    pcout << "//-----------------------------------------------" << std::endl;
    pcout << "Setting Matrix Sparsity Patterns................." << std::endl;
    pcout << "//-----------------------------------------------" << std::endl;

    stokes_matrix_zero_boundary.clear();

    stokes_matrix_liddriven_boundary.clear();

    TrilinosWrappers::SparsityPattern sp_zero (ns_index_set, MPI_COMM_WORLD);
    TrilinosWrappers::SparsityPattern sp_liddriven (ns_index_set, MPI_COMM_WORLD);

    DoFTools::make_sparsity_pattern (dof_handler,
                                         sp_zero,
                                     constraints_zero_bdry_conditions,
                                     true,  // keep constrained dofs
                                     Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));


    DoFTools::make_sparsity_pattern (dof_handler,
                                     sp_liddriven,
                                     constraints_liddriven_bdry_conditions,
                                     true,  // keep constrained dofs
                                     Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

    sp_zero.compress();
    sp_liddriven.compress();

    pcout << "//-----------------------------------------------" << std::endl;
    pcout << "Initializing Block Matrices and Vectors.........." << std::endl;
    pcout << "//-----------------------------------------------" << std::endl;

    stokes_matrix_liddriven_boundary.reinit(sp_liddriven);
    stokes_matrix_zero_boundary.reinit(sp_zero);

    system_solution_u_n.reinit(ns_relevant_set, MPI_COMM_WORLD);
    system_solution_y_n.reinit(ns_relevant_set, MPI_COMM_WORLD);
    system_solution_g_n.reinit(ns_relevant_set, MPI_COMM_WORLD);
    system_solution_w_n.reinit(ns_relevant_set, MPI_COMM_WORLD);
    system_solution_y_1_n.reinit(ns_relevant_set, MPI_COMM_WORLD);
    system_solution_y_2_n.reinit(ns_relevant_set, MPI_COMM_WORLD);

    // First index set is union of disjoint locally-owned index sets
    // across all processes.  Second index set is first set along
    // with ghost entries.  If Boolean variable is true then 
    // storage scheme for ghost elements allows multiple threads 
    // to write to ghost entries, if false then vector only has
    // read access to the relevant set containing ghost entries   
    system_rhs_zero_boundary.reinit(ns_index_set,
                                    ns_relevant_set,
                                    MPI_COMM_WORLD,
                                    true);
    system_rhs_liddriven_boundary.reinit(ns_index_set, 
                                         ns_relevant_set,
                                         MPI_COMM_WORLD,
                                         true);


  }


  template<int dim>
  void NavierStokesSolver<dim>::glowinski_initial_guess()
  {
    pcout << "//----------------------------------------------------------" << std::endl;
    pcout << "//Initializing Glowinski Systems and Initial Stokes Solve..." << std::endl;
    pcout << "//----------------------------------------------------------" << std::endl;
    // ------------------------------------
    // Step 1 - Initial Guess u^0
    // ------------------------------------
    
    // Step 1a - triangulation, dofs, matrix, and vector initialization
    setup_glowinski_systems(); 
    // Step 1b - FE system assemble for initial solve/iteration with LidDriven BCs
    assemble_stokes_matrix_and_rhs();
    // Step 1c - solve stokes system to obtain initial guess: u_0 in V_g
    {
      stokes_system_matrix.copy_from(stokes_matrix_liddriven_boundary);
      stokes_system_rhs = system_rhs_liddriven_boundary;
      solve_stokes_system("u_n");
    }
    // Step 1d - error check
    calculate_errors("u_n");

  }


  template<int dim>
  void NavierStokesSolver<dim>::glowinski_initial_solve_yn_gn_wn()
  {
    pcout << "//-------------------------------------------------" << std::endl;
    pcout << "//Solving for Initial y^{0}, g^{0}, and w^{0}......" << std::endl;
    pcout << "///------------------------------------------------" << std::endl;
    // ----------------------------------------------
    // Step 2 - Solve for y^{0}, g^{0}, and w^{0}
    // ----------------------------------------------
   
    // Step 2a - Solve Stokes system below for y^0 = y(u^0) in V_0
    //   laplace y^0 + grad sigma = -laplace u^0 + u^0 . grad u^0 - f
    //                    div y^0 = 0
    {
      assemble_stokes_y_n_system();
      stokes_system_matrix.copy_from(stokes_matrix_zero_boundary);
      stokes_system_rhs = system_rhs_zero_boundary;
      solve_stokes_system("y_n");
      calculate_errors("y_n");
    }

    // Step 2b - J_n relative error calculation's denominator
    a_of_y_0_y_0_value_on_process = 0.;
    a_of_y_0_y_0_value_on_process = calculate_bilinear_a_realnumber("y_n");
    a_of_y_0_y_0_value = 0.;
    a_of_y_0_y_0_value = 
               Utilities::MPI::sum(a_of_y_0_y_0_value_on_process, MPI_COMM_WORLD);
    pcout << "         a_of_y_0_y_0_value = " << a_of_y_0_y_0_value << std::endl;

    // Step 2c - Solve Stokes system below for g^0 in V_0
    //  -> a(g^0, z) + b(z, theta) = a(y^0, z) + c(u^0, z, y^0) + c(z, u^0, y^0)
    //                b(g^0, q)   = 0 
    {
      assemble_stokes_system("g_n");
      stokes_system_matrix.copy_from(stokes_matrix_zero_boundary);
      stokes_system_rhs = system_rhs_zero_boundary;
      solve_stokes_system("g_n");
      calculate_errors("g_n");
    }

    // Step 2d - g_n relative error calculation's denominator
    a_of_g_0_g_0_value_on_process = 0.;
    a_of_g_0_g_0_value_on_process = calculate_bilinear_a_realnumber("g_n");
    a_of_g_0_g_0_value = 0.;
    a_of_g_0_g_0_value = 
               Utilities::MPI::sum(a_of_g_0_g_0_value_on_process, MPI_COMM_WORLD);
    pcout << "         a_of_g_0_g_0_value = " << a_of_g_0_g_0_value << std::endl;
    // Step 2e - used for calculation of gamma_n
    a_of_g_n_g_n_value = 0.;
    a_of_g_n_g_n_value = a_of_g_0_g_0_value;

    // Step 2f - "calculation" of w^0 = g^0 (initial)
    system_solution_w_n = system_solution_g_n;

  }


  template<int dim>
  void NavierStokesSolver<dim>::glowinski_step(const TrilinosWrappers::MPI::Vector &src,
		                               TrilinosWrappers::MPI::Vector &dst)
  {
    pcout << "Taking step in glowinski algorithm..." << std::endl;

    system_solution_u_n = src;

    double g_n_rel_error = 1.;
    double gamma_n = 0.; 

    // Step 3a - Solve Stokes system (below) for y_1^n in V_0
    //    a(y_1^n, z) + b(z, theta) = a(w^n, z) + c(u^n, w^n, z) + c(w^n, u^n, z)
    //                b(y_1^n, q)   = 0 
    {
      assemble_stokes_system("y_1_n");
      stokes_system_matrix.copy_from(stokes_matrix_zero_boundary);
      stokes_system_rhs = system_rhs_zero_boundary;
      solve_stokes_system("y_1_n");
      calculate_errors("y_1_n");
    }
    // Step 3b - Solve Stokes system (below) for y_2^n in V_0
    //  -> a(y_2^n, z) + b(z, theta) = c(w^n, w^n, z) 
    //                 b(y_2^n, q)   = 0 
    {
      assemble_stokes_system("y_2_n");
      stokes_system_matrix.copy_from(stokes_matrix_zero_boundary);
      stokes_system_rhs = system_rhs_zero_boundary;
      solve_stokes_system("y_2_n");
      calculate_errors("y_2_n");
    }

    // Step 3c - Loop to minimize j_n'(rho) 
    pcout << "//------------------------------------------------------" << std::endl;
    pcout << "//Starting Newton Iteration to Determine new rho value..." << std::endl;
    pcout << "//------------------------------------------------------" << std::endl;
    // Newton's method to find zero rho_k of equation j_n'(rho_k) = 0
    // where j_n(rho_k) = 1/2 a(y^n(rho_k), y^n(rho_k)) 
    // and   y^n(rho_k) = y^n - rho_k y_1^n + rho_k^2 y_2^n  
    double rho_relative_error = 1.;
    unsigned int n_newton_iters = 1;
    double rho_value = 0.; double rho_value_old = 0.;
    double newton_stopping_criterion = std::pow(10.,-6);
    // zeroth, first and second derivative of j_n = j_n(rho)
    Tensor<1,3> derivatives_of_j_n_at_rho;
    while ( rho_relative_error > newton_stopping_criterion && n_newton_iters < 100)
    {
      // calculate zero, first and second derivatives of j_n(rho)
      derivatives_of_j_n_at_rho = calculate_derivs_of_j_n_at_rho(rho_value);
      rho_value = rho_value_old - derivatives_of_j_n_at_rho[1] / derivatives_of_j_n_at_rho[2];
      // calculate relative error and update old rho value with new value
      rho_relative_error = std::fabs(rho_value - rho_value_old)/std::fabs(rho_value);
      rho_value_old = rho_value;
      ++n_newton_iters;
    }
    pcout << "//-------------------------------------------------------" << std::endl;
    pcout << "//Newton iterations (for new rho value)  = " << n_newton_iters
	  << std::endl;
    pcout << "//relative error (for new rho value) = " << rho_relative_error 
	  << std::endl;
    pcout << "//-------------------------------------------------------" << std::endl;

    // Step 3d 
    pcout << "//------------------------------------------------------" << std::endl;
    pcout << "//Updating u^n and y^n ................................." << std::endl;
    pcout << "//------------------------------------------------------" << std::endl;
    {
      TrilinosWrappers::MPI::Vector system_solution_reuse_vec1 (ns_index_set,
                                                                MPI_COMM_WORLD);
      TrilinosWrappers::MPI::Vector system_solution_reuse_vec2 (ns_index_set,
                                                                MPI_COMM_WORLD);
      TrilinosWrappers::MPI::Vector system_solution_reuse_vec3 (ns_index_set,
                                                                MPI_COMM_WORLD);
      // u^n+1 = u^n - rho_n * w^n
      // Note: Location that caused incorrect results in undergraduate paper
      //       (sign error in the first update - corrected to subtracting rho_n*w^n)
      system_solution_reuse_vec1 = system_solution_u_n;
      system_solution_reuse_vec2 = system_solution_w_n;
      system_solution_reuse_vec1.add(-1.0*rho_value, system_solution_reuse_vec2);
      system_solution_u_n = system_solution_reuse_vec1;

      // y^n+1 = y^n(rho_n) = y^n - rho_n y_1^n + rho_n^2 y_2^n
      system_solution_reuse_vec1 = system_solution_y_n;
      system_solution_reuse_vec2 = system_solution_y_1_n;
      system_solution_reuse_vec3 = system_solution_y_2_n;
      system_solution_reuse_vec1.add(-rho_value, system_solution_reuse_vec2,
                                     rho_value*rho_value, system_solution_reuse_vec3);
      system_solution_y_n = system_solution_reuse_vec1;

      calculate_errors("u_n");
      calculate_errors("y_n");
    }

    // Step 3e - Solve Stokes system (below) for g^n+1 in V_0
    //     a(g^n+1, z) + b(z, theta) = a(y^n+1, z) + c(u^n+1, z, y^n+1) + c(z, u^n+1, y^n+1)
    //                 b(g^n+1, q)   = 0 
    {
      assemble_stokes_system("g_n");
      stokes_system_matrix.copy_from(stokes_matrix_zero_boundary);
      stokes_system_rhs = system_rhs_zero_boundary;
      solve_stokes_system("g_n");
      calculate_errors("g_n");
    }
   
    // Step 3f - Calculate a(g_n,g_n) and update gamma_n
    pcout << "//------------------------------------------------------" << std::endl;
    pcout << "//Updating gamma_n......................................" << std::endl;
    pcout << "//------------------------------------------------------" << std::endl;
    a_of_g_n_g_n_value_old = 0.;
    a_of_g_n_g_n_value_old = a_of_g_n_g_n_value;
    a_of_g_n_g_n_value_on_process = 0.; 
    a_of_g_n_g_n_value_on_process = calculate_bilinear_a_realnumber("g_n");
    a_of_g_n_g_n_value = 0.;
    a_of_g_n_g_n_value = 
                 Utilities::MPI::sum(a_of_g_n_g_n_value_on_process, MPI_COMM_WORLD);
    pcout << "         a_of_g_n_g_n_value = " << a_of_g_n_g_n_value << std::endl;

    // Calculate relative error a(g^{n+1},g^{n+1})/a(g^0,g^0)
    g_n_rel_error = a_of_g_n_g_n_value / a_of_g_0_g_0_value;
    pcout << "         g_n relative error = " << g_n_rel_error << std::endl;

    // Calculate gamma_n = a(g^{n+1},g^{n+1}) / a(g^n,g^n)
    gamma_n = a_of_g_n_g_n_value / a_of_g_n_g_n_value_old;

    // Step 3g - Set w^{n+1} = g^{n+1} + gamma_n * w^n
    pcout << "//------------------------------------------------------" << std::endl;
    pcout << "//Updating w^{n}........................................" << std::endl;
    pcout << "//------------------------------------------------------" << std::endl;
    {
      TrilinosWrappers::MPI::Vector system_solution_reuse_vec1 (ns_index_set,
                                                                MPI_COMM_WORLD);
      TrilinosWrappers::MPI::Vector system_solution_reuse_vec2 (ns_index_set,
                                                                MPI_COMM_WORLD);
      system_solution_reuse_vec1 = system_solution_w_n;
      system_solution_reuse_vec1 *= gamma_n;
      system_solution_reuse_vec2 = system_solution_g_n;
      system_solution_reuse_vec1 += system_solution_reuse_vec2;
      system_solution_w_n = system_solution_reuse_vec1;
    }

    // Step 3h - Calculate a(y_n,y_n) and relative error J(u^{n+1})/J(u^{0})
    a_of_y_n_y_n_value = 0.;
    a_of_y_n_y_n_value = derivatives_of_j_n_at_rho[0];
    J_n_rel_error = 0.;
    J_n_rel_error = a_of_y_n_y_n_value / a_of_y_0_y_0_value;
    pcout << "         J_n relative error = " << J_n_rel_error << std::endl;

    dst = system_solution_u_n; 
    
  }


  template<int dim>
  void NavierStokesSolver<dim>::run_glowinski_algorithm()
  {
    pcout << "//--------------------------------------" << std::endl;
    pcout << "//Starting glowinski algorithm.........." << std::endl;
    pcout << "//--------------------------------------" << std::endl;

    // Step 1
    glowinski_initial_guess();
    // Step 2
    glowinski_initial_solve_yn_gn_wn();

    // Step 3
    double stopping_criterion = std::pow(10.,-6);
    unsigned int n_minimization_loop_iters = 1;
    TrilinosWrappers::MPI::Vector u_previous(ns_index_set, MPI_COMM_WORLD);
    TrilinosWrappers::MPI::Vector u_new(ns_index_set, MPI_COMM_WORLD);
    u_new = system_solution_u_n;
    pcout << "//Entering (Outer) Minimization Loop - Minimizing J_n......." << std::endl;
    while (J_n_rel_error > stopping_criterion && n_minimization_loop_iters < 1000)
    {
      pcout << "//Minimization iteration number = " << n_minimization_loop_iters
	    << std::endl;
      u_previous = system_solution_u_n;
      glowinski_step(u_previous, u_new);
      ++n_minimization_loop_iters;
    }

    pcout << "*****************************************************" << std::endl;
    pcout << "***********Ending outer while loop to minimize J_n..." << std::endl;
    pcout << "*****************************************************" << std::endl;
    pcout << "         Number of J_n minimization iterations = " 
	  << n_minimization_loop_iters
          << " with relative error = " << J_n_rel_error << std::endl;

    // Step 4 - Post processing to solve for pressure and update system_solution_u_n
    pcout << "//------------------------------------------------------" << std::endl;
    pcout << "//Post processing to Obtain Pressure Solution..........." << std::endl;
    pcout << "//------------------------------------------------------" << std::endl;
    // Solve Stokes system (below) for y^{n+1} = y(u^{n+1}) in V_0 
    //   -> - laplace y^n+1 + grad sigma = - laplace u^n+1 + u^n+1 . grad u^n+1 - f 
    //                         div y^n+1 = 0
    // Really solving for the pressure sigma, 
    // since the negative of sigma is the pressure soln for Navier-Stokes equations
    {
      assemble_stokes_y_n_system();
      stokes_system_matrix.copy_from(stokes_matrix_zero_boundary);
      stokes_system_rhs = system_rhs_zero_boundary;
      solve_stokes_system("y_n");
      calculate_errors("y_n");
      
      TrilinosWrappers::MPI::Vector system_solution_reuse_vec1 (ns_index_set,
                                                                  MPI_COMM_WORLD);
      TrilinosWrappers::MPI::Vector system_solution_reuse_vec2 (ns_index_set,
                                                                  MPI_COMM_WORLD);
      system_solution_reuse_vec1 = system_solution_u_n;
      system_solution_reuse_vec2 = system_solution_y_n;

      const FEValuesExtractors::Scalar pressure(dim);
      IndexSet local_pressure_dofs = DoFTools::extract_dofs(dof_handler,
                                                            fe.component_mask(pressure));
      for (auto index = local_pressure_dofs.begin(); 
                  index != local_pressure_dofs.end(); ++index)
        system_solution_reuse_vec1(*index) = -system_solution_reuse_vec2(*index);

      system_solution_u_n = system_solution_reuse_vec1;

      // post process the pressure by subtracting away mean value
      post_process_pressure();

      calculate_errors("u_n");

    }

    output_data();

  }


//  template<int dim>
//  void NavierStokesSolver<dim>::run_glowinski_with_aa()
//  {
//    pcout << "Starting Glowinski Algorithm with Anderson Acceleration..."
//	  << std::endl;
//    pcout << "Number of mpi processes = " << n_mpi_processes << std::endl;
//
//    glowinski_initial_guess();
//
//  }



} // end LeastSquaresNavierStokes namespace


int main(int argc, char *argv[])
{

  try
  {
    using namespace dealii;
    using namespace LeastSquaresNavierStokes;

    deallog.depth_console(0);


    Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

    {
      const int dim = 2;
      NavierStokesSolver<dim> navier_stokes_problem;
      navier_stokes_problem.run_glowinski_algorithm();
    }

  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "-----------------------------------"
              << std::endl;
    std::cerr << "Exception on processings: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "-----------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "-----------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "-----------------------------------"
              << std::endl;
  } 

  return 0;
}





