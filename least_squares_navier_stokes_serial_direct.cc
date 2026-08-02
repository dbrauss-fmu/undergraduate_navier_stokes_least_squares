
/***************************************************************************************************
 * Stead State Two-Dimensional Serial Direct Navier-Stokes Least Squares Solver
 * 
 * Authors: Boatwright, Taylor
 *          Breeden, Ja'Nya
 *          Lytch, Jada
 *          Brauss, Daniel
 *          
 * 
 * Solves Stationary Navier-Stokes equations with UMFPack. 
 *
 * Contains an Exact Solution for Convergence Check.
 * 
 **************************************************************************************************/

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <cmath>


using namespace dealii;

/*******************************************************************************
 *Navier-Stokes 2D Exact Solution***********************************************
 *******************************************************************************
 *******************************************************************************
 *******************************************************************************
 ******************************************************************************/
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
  //  values(2) = -0.5*exp(2*x) + 0.5 + exp(-1.*y*y) - 1.; 

  // below the pressure mean value is subtracted away
  // the mean value of the pressure p = -0.5*exp(2x) + exp(-1.*y);
  //   int_{0}^{1} int_{0}^{1} p(x,y) dx dy
  //      = int_{0}^{1} [ -0.25*exp(2x) + x*exp(-y) ]_{x=0}^{x=1} dy
  //      = int_{0}^{1} -0.25*exp(2) + 0.25 + exp(-y) dy
  //      = [ (-0.25*exp(2) + 0.25)y - exp(-y) ]_{y=0}^{y=1}
  //      = -0.25*exp(2) + 0.25 - exp(-1) + 1
  //      = 1.25 - 0.25*exp(2) - exp(-1) 
  values(2) = -0.5*exp(2.*x) + exp(-1.*y)
                 - (1.25 - 0.25*exp(2.) - exp(-1.));	  
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
	
  values(0) = 0.;
  //values(1) = -p[1]*exp(-1.*p[1]*p[1]); 
  values(1) = -1.*exp(-1.*p[1]);
  values(2) = 0.;
	
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
  //values(2) = -0.5*exp(2*x) + 0.5 + exp(-1.*y*y) - 1.; 
  values(2) = -0.5*exp(2.*x) + exp(-1.*y)
                 - (1.25 - 0.25*exp(2.) - exp(-1.));	  
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
  //gradients[2][1] = -y*exp(-1.*y*y);
  gradients[2][1] = -1.*exp(-1.*y);

}

/*******************************************************************************
 * Navier-Stokes Solver using Least Squares and Conjugate Gradient Algorithms
 *******************************************************************************
 *******************************************************************************
 ******************************************************************************/
template<int dim>
class NavierStokesSolver
{
    public:
      NavierStokesSolver(unsigned int cells_per_coord_direction);              
      void run_least_squares_loop();    

    private:

      /* Member Functions */        
      void output_results();      
      void calculate_error(std::string solution_name);  
      void print_errors();       

      void assemble_stokes_system();      
      void solve_stokes();                
      void post_process_pressure();       
      void post_process_pressure_mean();  
      
      void setup_systems();	                // Step 1
      void assemble_stokes_y_n_system();	// Step 2
      void solve_y_n_system();
      void assemble_stokes_g_n_system();      	// Step 3
      void solve_g_n_system();
      void assemble_stokes_y_1_n_system();    	// Step 5
      void solve_y_1_n_system();
      void assemble_stokes_y_2_n_system();    	// Step 6
      void solve_y_2_n_system();
      Tensor<1,3> calculate_derivs_of_j_n_at_rho(double rho);
      double calculate_a_of_g_n_g_n();
      double calculate_a_of_y_n_y_n();

      /* Member Variables */        
      Triangulation<dim>     mesh;
      FESystem<dim>          fe;
      DoFHandler<dim>        dof_handler;
		
      AffineConstraints<double>       hanging_nodes_constraint_matrix; 

      ConvergenceTable	      velocity_convergence_table;
      ConvergenceTable	      pressure_convergence_table;
	
      const unsigned int     nx, ny;
      double                 nu, domain_length, domain_height;
      std::string            output_name;

      SparsityPattern        sparsity_pattern;
		
      Vector<double>         solution;
      
      Vector<double>         stokes_system_rhs;       // Step-1
      SparseMatrix<double>   stokes_system_matrix;    // Step-1

      SparseMatrix<double>   y_n_system_matrix;  
      Vector<double>         y_n_system_rhs; 
      Vector<double>         y_n_solution;
      
      SparseMatrix<double>   g_n_system_matrix;  
      Vector<double>         g_n_system_rhs; 
      Vector<double>         g_n_solution;

      Vector<double>         w_n_solution;

      SparseMatrix<double>   y_1_n_system_matrix;  
      Vector<double>         y_1_n_system_rhs; 
      Vector<double>         y_1_n_solution;

      SparseMatrix<double>   y_2_n_system_matrix;  
      Vector<double>         y_2_n_system_rhs; 
      Vector<double>         y_2_n_solution;

};


/***********************************************************************************************
 *  taylor-hood elements, quadratic basis functions for 2 velocity components, 
 * linear basis functions for pressure
 **********************************************************************************************/
template<int dim>
NavierStokesSolver<dim>::NavierStokesSolver(unsigned int cells_per_coord_direction)
    :
    fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1),
    dof_handler(mesh),
    nx(cells_per_coord_direction),
    ny(cells_per_coord_direction),
    nu(1.00),
    domain_length(1.),
    domain_height(1.),
    output_name("navier_stokes")
{}



// *****************************************************************************
// Step 14f - Post Processing the Pressure
// *****************************************************************************
template<int dim>
void NavierStokesSolver<dim>::post_process_pressure()
{
  printf("Post-Processing Pressure...\n");
  
  //set boundary labels:
  // (1) : x = 0
  // (2) : x = domain_length = 1
  // (3) : y = domain_height = 1
  // (4) : y = 0
  typename DoFHandler<dim>::active_cell_iterator  
			cell = dof_handler.begin_active(), endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; f++)
      if (cell->face(f)->at_boundary())
        if (fabs(cell->face(f)->center()[0]) < 1.0e-8)
          cell->face(f)->set_boundary_id(1);


  // pressure at x = 0 equals zero according to exact solution
  // we determine the pressure at x = 0 and adjust all pressure solutions
  // by subtracting this value
  std::vector<bool> x_equal_zero_press_dofs (dof_handler.n_dofs(), false);
  std::vector<bool> press_select(dim+1, false);
  press_select[dim] = true;

  std::set<unsigned int> bdy_indic_x_equal_zero;
  unsigned int bdy_x_equal_0 = 1;
  bdy_indic_x_equal_zero.insert(bdy_x_equal_0);

  DoFTools::extract_boundary_dofs (dof_handler,
                                         press_select,
                                         x_equal_zero_press_dofs, // output
                                         bdy_indic_x_equal_zero);
  // first pressure dof where x = 0
  const unsigned int first_x_equal_zero_press_dof_index               
          = std::distance (x_equal_zero_press_dofs.begin(),
                           std::find (x_equal_zero_press_dofs.begin(),
                                      x_equal_zero_press_dofs.end(),
                                      true));
                                      
  double pressure_value = solution(first_x_equal_zero_press_dof_index);
  std::cout << "pressure_value = " << pressure_value << " = solution(" 
            << first_x_equal_zero_press_dof_index << ")" << std::endl;

  const FEValuesExtractors::Scalar pressure(dim);
  const IndexSet pressure_dofs = DoFTools::extract_dofs(dof_handler,
	                                                fe.component_mask(pressure));
  auto it = pressure_dofs.begin();
  auto end_it = pressure_dofs.end();
  for (; it != end_it; ++it)
  {
    dealii::types::global_dof_index index = *it;
    solution(index) -= pressure_value;
  }

}


template<int dim>
void NavierStokesSolver<dim>::post_process_pressure_mean()
{
  printf("Averaging Post-Process for Pressure...\n");

  const double pressure_mean_value = VectorTools::compute_mean_value (dof_handler,
                                                                      QGauss<dim>(3),
                                                                      solution,
                                                                      dim);
  std::cout << "pressure_mean_value = " << pressure_mean_value << std::endl;
  double neg_pressure_mean_value = -1.0*pressure_mean_value;

  const FEValuesExtractors::Scalar pressure(dim);
  const IndexSet pressure_dofs = DoFTools::extract_dofs(dof_handler,
	                                                fe.component_mask(pressure));
  auto it = pressure_dofs.begin();
  auto end_it = pressure_dofs.end();
  for (; it != end_it; ++it)
  {
    dealii::types::global_dof_index index = *it;
    solution(index) += neg_pressure_mean_value;
  }

}


// *****************************************************************************
// Step 1b - Setting Up Stokes System for Initial Guess
// *****************************************************************************
// Second call in running the algorithm.
// The initial guess for the algorithm comes from a Stokes solve for the given (steady state) boundary conditions.
// Therefore, this member function sets up the stokes system to be solved.
template<int dim>
void NavierStokesSolver<dim>::assemble_stokes_system()
{
  Timer timer;
  timer.start ();

  ExactSolutionForcingFunction<dim>  forcing_function;		
	
  QGauss<dim>                 quadrature_formula(2);
  QGauss<dim-1>               face_quadrature_formula(3);
	
  const int                   dofs_per_cell = fe.dofs_per_cell;
  const int                   n_q_points = quadrature_formula.size();	

  std::vector<Vector<double> >       rhs_values (n_q_points, Vector<double>(dim+1));
	
  std::vector<Tensor<2,dim> > grad_phi_u (dofs_per_cell);
  std::vector<double>         div_phi_u (dofs_per_cell);
  std::vector<double>         phi_p (dofs_per_cell);
  std::vector<Tensor<1,dim> > phi_u (dofs_per_cell);
  Vector<double>              cell_rhs(dofs_per_cell);
	
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
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_mass   = 0;
    cell_rhs = 0;

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
			  (
				fe_values.shape_value(i,q)*rhs_values[q](equation_i)         
			  )
					*fe_values.JxW(q);
        
      } // loop over rows
      
    }  // loop over quadrature points q
		
    cell->get_dof_indices(local_dof_indices);     
    hanging_nodes_constraint_matrix.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, 
					   	                 stokes_system_matrix, stokes_system_rhs);      
    
  } // loop over cells	
    
  // interpolating boundary values for velocities
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             ExactSolutionBoundaryValues<dim>(),
                                             boundary_values,
                                             fe.component_mask(velocities));
  MatrixTools::apply_boundary_values(boundary_values,
                                       stokes_system_matrix,
                                       solution,
                                       stokes_system_rhs);
    
    
  timer.stop();

}



// ************************************************************************************
// Step 1c - Solving Stokes System Directly using UMFPACK for Algorithm Initial Guess 
// ************************************************************************************
template<int dim>
void NavierStokesSolver<dim>::solve_stokes()
{
  printf("Solving Stokes System for Version 2 of Newton's Method...\n");
  Timer timer;
  timer.start ();
	
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(stokes_system_matrix);

  A_direct.vmult(solution, stokes_system_rhs);

  hanging_nodes_constraint_matrix.distribute(solution); 

  timer.stop ();

}


// *****************************************************************************
// Step 14i - Outputting Results
// *****************************************************************************
template<int dim>
void NavierStokesSolver<dim>::output_results()
{	
	std::vector<std::string> solution_names;
	solution_names.push_back("u");
	solution_names.push_back("v");
	solution_names.push_back("pressure");
			
	DataOut<2> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, solution_names);
			
	data_out.build_patches ();
	std::ostringstream filename;
	filename << output_name << "-" << nu << ".vtk";
	std::ofstream output (filename.str().c_str());
	data_out.write_vtk (output);


    // coming from deal.ii step-22	
    std::vector<std::string> solution_names_part2(dim, "velocity");        
    solution_names_part2.emplace_back("pressure");                         
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out_part2;                                           
    data_out_part2.attach_dof_handler(dof_handler);                        
    data_out_part2.add_data_vector(solution,                      
                             solution_names_part2,                         
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out_part2.build_patches();                                        
    std::ostringstream filename_part2;
    filename_part2 << output_name << "-solution-with-vectors-" << nu << ".vtk";
    std::ofstream output_part2(filename_part2.str().c_str());              
    data_out_part2.write_vtk(output_part2);	                           
	
}


// *****************************************************************************
// The Error Calculation is Used in the Following Steps for the Least Squares Algorithm
// Step 1d     - Calculating Error from Stokes Solve Initial Guess u^0
// Step 2c     - Calculating Error from Stokes System for y^0
// Step 5c     - Calculating Error from Stokes System for y_1^n
// Step 6c     - Calculating Error from Stokes System for y_2^n
// Step 8c, 8e - Calculating Error from Stokes System for u^n, y^n
// Step 14d, 14g - Calculating Error from Stokes System for u^n (before/after pressure post-process)
// *****************************************************************************
template<int dim>
void NavierStokesSolver<dim>::calculate_error(std::string solution_name)
{
  ExactSolution<dim>  exact_solution;	
  
  const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0,dim),3);
  const ComponentSelectFunction<dim> pressure_mask(dim,3);
	
  Vector<float> difference_per_cell (mesh.n_active_cells());

  if (solution_name == "u_n")
  {
    /**Calculate velocity errors************************/
    VectorTools::integrate_difference (dof_handler, solution, exact_solution, difference_per_cell,
                                       QGauss<dim>(3), VectorTools::L2_norm, &velocity_mask);
    const double L2_error_velocity = difference_per_cell.l2_norm();
    
    VectorTools::integrate_difference (dof_handler, solution, exact_solution, difference_per_cell,
					 QGauss<dim>(3), VectorTools::H1_seminorm, &velocity_mask);
    const double H1_error_velocity = difference_per_cell.l2_norm();

    /** Calculate pressure errors***********************/
    VectorTools::integrate_difference (dof_handler, solution, exact_solution, difference_per_cell, 
						QGauss<dim>(3), VectorTools::L2_norm, &pressure_mask);
    const double L2_error_pressure = difference_per_cell.l2_norm();
    
    VectorTools::integrate_difference (dof_handler, solution, exact_solution, difference_per_cell,
						QGauss<dim>(3), VectorTools::H1_seminorm, &pressure_mask);
    const double H1_error_pressure = difference_per_cell.l2_norm();

    std::cout << "u velocity L2 error for " << solution_name
              << " is : " << L2_error_velocity << std::endl;
    std::cout << "u velocity H1 error for " << solution_name
              << " is : " << H1_error_velocity << std::endl;
    std::cout << "p pressure L2 error for " << solution_name
              << " is : " << L2_error_pressure << std::endl;
    std::cout << "p pressure H1 error for " << solution_name
              << " is : " << H1_error_pressure << std::endl;

    velocity_convergence_table.add_value("cells", mesh.n_active_cells());
    velocity_convergence_table.add_value("dofs", dof_handler.n_dofs());
    velocity_convergence_table.add_value("L2", L2_error_velocity);
    velocity_convergence_table.add_value("H1", H1_error_velocity);
	
    pressure_convergence_table.add_value("cells", mesh.n_active_cells());
    pressure_convergence_table.add_value("dofs", dof_handler.n_dofs());
    pressure_convergence_table.add_value("L2", L2_error_pressure);
    pressure_convergence_table.add_value("H1", H1_error_pressure);
  }
  else if (solution_name == "g_n")
  {
    /**Calculate velocity errors************************/
    VectorTools::integrate_difference (dof_handler, g_n_solution, exact_solution, difference_per_cell,
                                       QGauss<dim>(3), VectorTools::L2_norm, &velocity_mask);
    const double L2_error_velocity = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (dof_handler, g_n_solution, exact_solution, difference_per_cell,
                                         QGauss<dim>(3), VectorTools::H1_seminorm, &velocity_mask);
    const double H1_error_velocity = difference_per_cell.l2_norm();

    /** Calculate pressure errors***********************/
    VectorTools::integrate_difference (dof_handler, g_n_solution, exact_solution, difference_per_cell,
                                                QGauss<dim>(3), VectorTools::L2_norm, &pressure_mask);
    const double L2_error_pressure = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (dof_handler, g_n_solution, exact_solution, difference_per_cell,
                                                QGauss<dim>(3), VectorTools::H1_seminorm, &pressure_mask);
    const double H1_error_pressure = difference_per_cell.l2_norm();

    std::cout << "u velocity L2 error for " << solution_name
              << " is : " << L2_error_velocity << std::endl;
    std::cout << "u velocity H1 error for " << solution_name
              << " is : " << H1_error_velocity << std::endl;
    std::cout << "p pressure L2 error for " << solution_name
              << " is : " << L2_error_pressure << std::endl;
    std::cout << "p pressure H1 error for " << solution_name
              << " is : " << H1_error_pressure << std::endl;
  }
  else if (solution_name == "y_n")
  {
    /**Calculate velocity errors************************/
    VectorTools::integrate_difference (dof_handler, y_n_solution, exact_solution, difference_per_cell,
                                       QGauss<dim>(3), VectorTools::L2_norm, &velocity_mask);
    const double L2_error_velocity = difference_per_cell.l2_norm();
    
    VectorTools::integrate_difference (dof_handler, y_n_solution, exact_solution, difference_per_cell,
					 QGauss<dim>(3), VectorTools::H1_seminorm, &velocity_mask);
    const double H1_error_velocity = difference_per_cell.l2_norm();

    /** Calculate pressure errors***********************/
    VectorTools::integrate_difference (dof_handler, y_n_solution, exact_solution, difference_per_cell, 
						QGauss<dim>(3), VectorTools::L2_norm, &pressure_mask);
    const double L2_error_pressure = difference_per_cell.l2_norm();
    
    VectorTools::integrate_difference (dof_handler, y_n_solution, exact_solution, difference_per_cell,
						QGauss<dim>(3), VectorTools::H1_seminorm, &pressure_mask);
    const double H1_error_pressure = difference_per_cell.l2_norm();

    std::cout << "u velocity L2 error for " << solution_name
              << " is : " << L2_error_velocity << std::endl;
    std::cout << "u velocity H1 error for " << solution_name
              << " is : " << H1_error_velocity << std::endl;
    std::cout << "p pressure L2 error for " << solution_name
              << " is : " << L2_error_pressure << std::endl;
    std::cout << "p pressure H1 error for " << solution_name
              << " is : " << H1_error_pressure << std::endl;
  }
  else if (solution_name == "y_1_n")
  {
    /**Calculate velocity errors************************/
    VectorTools::integrate_difference (dof_handler, y_1_n_solution, exact_solution, difference_per_cell,
                                       QGauss<dim>(3), VectorTools::L2_norm, &velocity_mask);
    const double L2_error_velocity = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (dof_handler, y_1_n_solution, exact_solution, difference_per_cell,
                                         QGauss<dim>(3), VectorTools::H1_seminorm, &velocity_mask);
    const double H1_error_velocity = difference_per_cell.l2_norm();

    /** Calculate pressure errors***********************/
    VectorTools::integrate_difference (dof_handler, y_1_n_solution, exact_solution, difference_per_cell,
                                                QGauss<dim>(3), VectorTools::L2_norm, &pressure_mask);
    const double L2_error_pressure = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (dof_handler, y_1_n_solution, exact_solution, difference_per_cell,
                                                QGauss<dim>(3), VectorTools::H1_seminorm, &pressure_mask);
    const double H1_error_pressure = difference_per_cell.l2_norm();

    std::cout << "u velocity L2 error for " << solution_name
              << " is : " << L2_error_velocity << std::endl;
    std::cout << "u velocity H1 error for " << solution_name
              << " is : " << H1_error_velocity << std::endl;
    std::cout << "p pressure L2 error for " << solution_name
              << " is : " << L2_error_pressure << std::endl;
    std::cout << "p pressure H1 error for " << solution_name
              << " is : " << H1_error_pressure << std::endl;
  }
  else if (solution_name == "y_2_n")
  {
    /**Calculate velocity errors************************/
    VectorTools::integrate_difference (dof_handler, y_2_n_solution, exact_solution, difference_per_cell,
                                       QGauss<dim>(3), VectorTools::L2_norm, &velocity_mask);
    const double L2_error_velocity = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (dof_handler, y_2_n_solution, exact_solution, difference_per_cell,
                                         QGauss<dim>(3), VectorTools::H1_seminorm, &velocity_mask);
    const double H1_error_velocity = difference_per_cell.l2_norm();

    /** Calculate pressure errors***********************/
    VectorTools::integrate_difference (dof_handler, y_2_n_solution, exact_solution, difference_per_cell,
                                                QGauss<dim>(3), VectorTools::L2_norm, &pressure_mask);
    const double L2_error_pressure = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (dof_handler, y_2_n_solution, exact_solution, difference_per_cell,
                                                QGauss<dim>(3), VectorTools::H1_seminorm, &pressure_mask);
    const double H1_error_pressure = difference_per_cell.l2_norm();

    std::cout << "u velocity L2 error for " << solution_name
              << " is : " << L2_error_velocity << std::endl;
    std::cout << "u velocity H1 error for " << solution_name
              << " is : " << H1_error_velocity << std::endl;
    std::cout << "p pressure L2 error for " << solution_name
              << " is : " << L2_error_pressure << std::endl;
    std::cout << "p pressure H1 error for " << solution_name
              << " is : " << H1_error_pressure << std::endl;
  }
  else
  {
    Assert(1==0, ExcNotImplemented());
  }

}


// ***************************************************************************************
// The Error Printing is Used in the Following Steps for the Least Squares Algorithm
// Step 8d         - Printing Error of Update u^{n+1}
// Step 14e, 14h   - Printing Error of Update u^{n+1} (before/after pressure post process
// ***************************************************************************************
template<int dim>
void NavierStokesSolver<dim>::print_errors()
{
	velocity_convergence_table.set_precision("L2", 3);
	velocity_convergence_table.set_precision("H1", 3);
	velocity_convergence_table.set_scientific("L2", true);
	velocity_convergence_table.set_scientific("H1", true);
	
	velocity_convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
	velocity_convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);

	pressure_convergence_table.set_precision("L2", 3);
	pressure_convergence_table.set_precision("H1", 3);
	pressure_convergence_table.set_scientific("L2", true);
	pressure_convergence_table.set_scientific("H1", true);
	
	pressure_convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
	pressure_convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);
	
	printf("\n");
	printf("Velocity error\n");
	velocity_convergence_table.write_text(std::cout);
	
	printf("\n");
	printf("Pressure error\n");
	pressure_convergence_table.write_text(std::cout);
}


 /* least squares solver - based on page 81 of Multidisciplinary Methods 
  *                             and page 92 of Finite Element Methods by Gunzburger
  *                             and page 88 of Numerical Methods by Bristeau
 */ 
// NOTE: ExactSolution Forcing Function utilized in the Steps
//       - Step 1b: assemble_stokes_system (ExactSolution Used in Setting Boundary Values also)
//       - Step 2a: assemble_stokes_y_n_system (ZeroFunction Used in Setting Boundary Values)
template<int dim>
void NavierStokesSolver<dim>::run_least_squares_loop()
{
  std::cout << "Starting Least Squares minimization loop..." << std::endl;

  // Step 1 - Initial Guess u^0
  // Solve Stokes Problem for u^0 in V_g 
  //   using exact solution boundary conditions and right-hand side 

  // Step 1a
  setup_systems();
  // Step 1b
  assemble_stokes_system();
  // Step 1c
  solve_stokes();
  // Step 1d
  calculate_error("u_n");
  
  // Step 2 - Solve for y^0 = y(u^0) in V_0
  //  -> - laplace y^0 + grad sigma = - laplace u^0 + u^0 . grad u^0 - f 
  //                        div y^0 = 0
  // Step 2a
  assemble_stokes_y_n_system();
  // Step 2b
  solve_y_n_system();
  // Step 2c
  calculate_error("y_n");
  // Step 2d - for relative error
  double a_of_y_0_y_0_value = calculate_a_of_y_n_y_n();
  std::cout << "a_of_y_0_y_0_value = " << a_of_y_0_y_0_value << std::endl;
  
  // Step 3 - Solve for g^0 in V_0
  //  -> a(g^0, z) + b(z, theta) = a(y^0, z) + c(u^0, z, y^0) + c(z, u^0, y^0)
  //                b(g^0, q)   = 0 
  // Step 3a
  assemble_stokes_g_n_system();
  // Step 3b
  solve_g_n_system();
  // Step 3c
  calculate_error("g_n");
  // Step 3d - for relative error
  double a_of_g_0_g_0_value = calculate_a_of_g_n_g_n();
  std::cout << "a_of_g_0_g_0_value = " << a_of_g_0_g_0_value << std::endl;
    
  // Step 4 - Set w^0 = g^0
  w_n_solution = g_n_solution;

  // Initializing variables for Least Squares Outer While Loop
  double g_n_rel_error = 1.;
  double J_n_rel_error = 1.;
  double stopping_criterion = std::pow(10.,-6);
  unsigned int minimization_iters = 1;
  double a_of_g_n_g_n_value = a_of_g_0_g_0_value;
  double a_of_y_n_y_n_value = 0;
  double a_of_g_n_g_n_value_old = 0;
  double gamma_n = 0;
  
  // Begin loop
  // Given u^n, g^n, w^n
  // Determine u^{n+1}, g^{n+1}, w^{n+1}
  while ( J_n_rel_error > stopping_criterion && minimization_iters < 100000)
            //&& g_n_rel_error > stopping_criterion)
  {
    
    // Step 5 - Solve y_1^n in V_0
    //  -> a(y_1^n, z) + b(z, theta) = a(w^n, z) + c(u^n, w^n, z) + c(w^n, u^n, z)
    //                 b(y_1^n, q)   = 0 
    // Step 5a
    assemble_stokes_y_1_n_system();
    // Step 5b
    solve_y_1_n_system();
    // Step 5c
    calculate_error("y_1_n");

    // Step 6 - Solve y_2^n in V_0
    //  -> a(y_2^n, z) + b(z, theta) = c(w^n, w^n, z) 
    //                 b(y_2^n, q)   = 0 
    // Step 6a
    assemble_stokes_y_2_n_system();
    // Step 6b
    solve_y_2_n_system();  
    // Step 6c
    calculate_error("y_2_n");

    // Step 7 - Apply Newton's Method to determine the real number rho_n
    //          that causes the derivative j_n'(rho_n) = 0 
    //          where
    //                  j_n(rho) = 1/2 a(y^n(rho), y^n(rho))
    //          and
    //                  y^n(rho) = y^n - rho y_1^n + rho^2 y_2^n
    //    
    // Initializing variables for Newton While Loop
    double newton_rel_error = 1.0;
    unsigned int newton_iters = 1;
    Tensor<1,3> derivatives_of_j_n_at_rho;
    double rho_value = 0.;  double rho_value_old = 0;

    while ( newton_rel_error > stopping_criterion && newton_iters < 100 )
    {
      // Step 7a
      derivatives_of_j_n_at_rho = calculate_derivs_of_j_n_at_rho(rho_value);
      // Step 7b - rho value calculation
      rho_value = rho_value_old - derivatives_of_j_n_at_rho[1] / derivatives_of_j_n_at_rho[2];
      // Step 7c - newton relative error calculation
      newton_rel_error = std::fabs(rho_value-rho_value_old)/std::fabs(rho_value);
      // Step 7d - updates for next pass of loop
      rho_value_old = rho_value;
      newton_iters++;
    }

    std::cout << "         Number of Newton iterations = " << newton_iters
          << " with newton relative error = " << newton_rel_error << std::endl;

  
    // Step 8 - Set u^n+1 = u^n - rho_n w^n
    //              y^n+1 = y^n(rho_n) = y^n - rho_n y_1^n + rho_n^2 y_2^n
    // This appears to be the location (a sign error in the first update (addition of rho_n w^n instead of subtraction)
    // that caused incorrect results in the undergraduates student paper
    // Step 8a - New solution u^{n+1} Calculation
    solution.add(-1.0*rho_value, w_n_solution);
    // Step 8b - New iteration value y^{n+1} Calculation
    y_n_solution.add(-rho_value, y_1_n_solution, rho_value*rho_value, y_2_n_solution);

    // Step 8c
    calculate_error("u_n");
    // Step 8d
    print_errors();	
    // Step 8e
    calculate_error("y_n");

    // Step 9 - Solve for g^n+1 in V_0
    //  -> a(g^n+1, z) + b(z, theta) = a(y^n+1, z) + c(u^n+1, z, y^n+1) + c(z, u^n+1, y^n+1)
    //                 b(g^n+1, q)   = 0 
    // Step 9a
    assemble_stokes_g_n_system();
    // Step 9b
    solve_g_n_system();
    // Step 9c - Update for gamma_n calculation
    a_of_g_n_g_n_value_old = a_of_g_n_g_n_value;
    // Step 9d
    a_of_g_n_g_n_value = calculate_a_of_g_n_g_n();

    std::cout << "a_of_g_n_g_n_value = " << a_of_g_n_g_n_value << std::endl;
    
    // Step 10 - Calculate rel_err_1 = a(g^n+1, g^n+1) / a(g^0, g^0)
    g_n_rel_error = a_of_g_n_g_n_value / a_of_g_0_g_0_value;
    std::cout << "g_n relative error = " << g_n_rel_error << std::endl;
    
    // Step 11 - Calculate gamma_n = a(g^n+1,g^n+1) / a(g^n,g^n)
    gamma_n = a_of_g_n_g_n_value / a_of_g_n_g_n_value_old;
    
    // Step 12 - Set w^n+1 = g^n+1 + gamma_n w^n   
    w_n_solution *= gamma_n;
    w_n_solution += g_n_solution;
    
    // Step 13 - Calculate the J_n relative_error =  J(u^n+1) / J(u^0)
    a_of_y_n_y_n_value = derivatives_of_j_n_at_rho[0];
    J_n_rel_error = a_of_y_n_y_n_value / a_of_y_0_y_0_value;
    std::cout << "J_n relative error = " << J_n_rel_error << std::endl;

    // Step 14 - Solve for y^n+1 = y(u^n+1) in V_0 and (really) for the pressure sigma
    //   -> - laplace y^n+1 + grad sigma = - laplace u^n+1 + u^n+1 . grad u^n+1 - f 
    //                         div y^n+1 = 0
    if (J_n_rel_error < stopping_criterion) // || g_n_rel_error < stopping_criterion)
    {
      std::cout << "The relative error for J is smaller than the stopping criterion..." << std::endl;
      // Step 14a
      assemble_stokes_y_n_system();
      // Step 14b
      solve_y_n_system();
      // Step 14c - Update to Pressure Solution (error tolerance has been met - finalizing solution)
      const FEValuesExtractors::Scalar pressure(dim);
      const IndexSet pressure_dofs = DoFTools::extract_dofs(dof_handler,
		                                            fe.component_mask(pressure));
      auto it = pressure_dofs.begin();
      auto end_it = pressure_dofs.end();
      for (; it != end_it; ++it)
      {
	dealii::types::global_dof_index index = *it;
        solution(index) = -y_n_solution(index);
      }
      // Step 14d
      calculate_error("u_n");
      // Step 14e
      print_errors();	
      // Step 14f
      //post_process_pressure();  
      post_process_pressure_mean();  
      // Step 14g
      calculate_error("u_n");
      // Step 14h
      print_errors();	
      // Step 14i
      output_results();

    } // end if - J_n stopping criterion

    // Step 15 - Updating for Next Loop Pass 
    minimization_iters++;
    std::cout << "minimization iterations = " << minimization_iters << std::endl;
    
  } // End while minimization loop

} 


// *****************************************************************************
// Step 1a - Grid Initialization and Matrix Systems Initialization
// *****************************************************************************
// First step in running the least squares algorithm.  nx and ny (set in constructor) are the number of nodes in the 
// x,y-directions, respectively.  Therefore, there are nx-1 and ny-1 FEM cells in the respective coordinate 
// direcitons, forming a 2D rectangular domain.  Here, the matrices and vectors that will be used in the 
// algorithm are initialized.
template<int dim>
void NavierStokesSolver<dim>::setup_systems()
{
  printf("Setting Up Mesh, Matrices and Vectors for Least Squares Method...\n");
  
  Timer timer;
  timer.start ();

  std::vector<unsigned int> number_elements(2);
  number_elements[0] = nx-1;
  number_elements[1] = ny-1;
	  
  Point<dim> bottom_left, top_right;
  bottom_left[0] = 0;  bottom_left[1] = 0;
  top_right[0] = 1;  top_right[1] = 1;
			
  GridGenerator::subdivided_hyper_rectangle(mesh, number_elements,
                                            bottom_left, top_right, false);
  
  dof_handler.distribute_dofs(fe);  
  
  std::printf("Number of active cells:%d\n", mesh.n_active_cells());
  std::printf("Number of degrees of freedom:%d\n", dof_handler.n_dofs()); 


  hanging_nodes_constraint_matrix.clear(); 
  DoFTools::make_hanging_node_constraints(dof_handler, hanging_nodes_constraint_matrix);
  hanging_nodes_constraint_matrix.close(); 


  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  hanging_nodes_constraint_matrix.condense(dsp);      
  
  sparsity_pattern.copy_from(dsp);

  stokes_system_matrix.reinit(sparsity_pattern);  
  stokes_system_rhs.reinit(dof_handler.n_dofs()); 
  
  solution.reinit(dof_handler.n_dofs());

  y_n_system_matrix.reinit(sparsity_pattern);  
  y_n_system_rhs.reinit(dof_handler.n_dofs()); 
  y_n_solution.reinit(dof_handler.n_dofs());

  g_n_system_matrix.reinit(sparsity_pattern);  
  g_n_system_rhs.reinit(dof_handler.n_dofs()); 
  g_n_solution.reinit(dof_handler.n_dofs());

  w_n_solution.reinit(dof_handler.n_dofs());

  y_1_n_system_matrix.reinit(sparsity_pattern);  
  y_1_n_system_rhs.reinit(dof_handler.n_dofs()); 
  y_1_n_solution.reinit(dof_handler.n_dofs());
  
  y_2_n_system_matrix.reinit(sparsity_pattern);  
  y_2_n_system_rhs.reinit(dof_handler.n_dofs()); 
  y_2_n_solution.reinit(dof_handler.n_dofs());
  
  timer.stop ();

}



// *****************************************************************************
// Step 2a, 14a - Setting up the system for y^n 
// *****************************************************************************
// Setting up the system to solve for y^n = y(u^n) in V_0
//  -> - laplace y^n + grad sigma = - laplace u^n + u^n . grad u^n - f 
//                        div y^n = 0
template<int dim>
void NavierStokesSolver<dim>::assemble_stokes_y_n_system()
{
  printf("Assembling Stokes y^n System for Least Squares Method...\n");

  Timer timer;
  timer.start ();

  y_n_system_matrix = 0;    
  y_n_system_rhs = 0;

  ExactSolutionForcingFunction<dim>  forcing_function;		
	
  QGauss<dim>                   quadrature_formula(2);
  QGauss<dim-1>                 face_quadrature_formula(3);
	
  const int                     dofs_per_cell = fe.dofs_per_cell;
  const int                     n_q_points = quadrature_formula.size();	

  std::vector<Vector<double> >  rhs_values (n_q_points, Vector<double>(dim+1));

  // current solution values
  std::vector<Tensor<1,dim> > 	solution_velocity_values(n_q_points);
  std::vector<Tensor<2,dim> > 	solution_velocity_gradients(n_q_points);
	
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
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_mass   = 0;
    cell_rhs = 0;

    /** Calculate velocity values and gradients from previous newton iteration
      * at each quadrature point in cell *********************************/
    fe_values[velocities].get_function_values(solution, solution_velocity_values);        
    fe_values[velocities].get_function_gradients(solution, solution_velocity_gradients);
    
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
		
    cell->get_dof_indices(local_dof_indices);    
    hanging_nodes_constraint_matrix.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, 
					   	                 y_n_system_matrix, y_n_system_rhs);      
    
  } // loop over cells	
    
  // interpolating boundary values for velocities
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(3),
                                             boundary_values,
                                             fe.component_mask(velocities));
  MatrixTools::apply_boundary_values(boundary_values,
                                       y_n_system_matrix,
                                       y_n_solution,
                                       y_n_system_rhs);
    
  timer.stop();

}



// *****************************************************************************
// Step 2b, 14b - Directly Solving for y^n using UMFPACK 
// *****************************************************************************
// Solving for y^n = y(u^n) in V_0
//  -> - laplace y^n + grad sigma = - laplace u^n + u^n . grad u^n - f 
//                        div y^n = 0
template<int dim>
void NavierStokesSolver<dim>::solve_y_n_system()
{
  printf("Solving Stokes y^n System for Least Squares Method...\n");
  Timer timer;
  timer.start ();
	
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(y_n_system_matrix);

  A_direct.vmult(y_n_solution, y_n_system_rhs);

  hanging_nodes_constraint_matrix.distribute(y_n_solution); 

  timer.stop ();

}


// *****************************************************************************
// Step 3a - Setting up the system for g^n 
// *****************************************************************************
// Step 3 - Setting up the System for g^0 in V_0
//  -> a(g^0, z) + b(z, theta) = a(y^0, z) + c(u^0, z, y^0) + c(z, u^0, y^0)
//                b(g^0, q)   = 0 
template<int dim>
void NavierStokesSolver<dim>::assemble_stokes_g_n_system()
{
  printf("Assembling Stokes g^n System for Least Squares Method...\n");

  Timer timer;
  timer.start ();

  g_n_system_matrix = 0;    
  g_n_system_rhs = 0;

  ExactSolutionForcingFunction<dim>  forcing_function;		
  
  QGauss<dim>                   quadrature_formula(2);
  QGauss<dim-1>                 face_quadrature_formula(3);
	
  const int                     dofs_per_cell = fe.dofs_per_cell;
  const int                     n_q_points = quadrature_formula.size();	

  std::vector<Vector<double> >  rhs_values (n_q_points, Vector<double>(dim+1));

  std::vector<Tensor<1,dim> > 	solution_velocity_values(n_q_points);
  std::vector<Tensor< 2, dim> > 	solution_velocity_gradients(n_q_points);
  std::vector<Tensor<1,dim> > 	y_n_velocity_values(n_q_points);
  std::vector<Tensor< 2, dim> > 	y_n_velocity_gradients(n_q_points);
	
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
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_mass   = 0;
    cell_rhs = 0;

    fe_values[velocities].get_function_values(solution, solution_velocity_values);        
    fe_values[velocities].get_function_gradients(solution, solution_velocity_gradients);
    fe_values[velocities].get_function_values(y_n_solution, y_n_velocity_values);        
    fe_values[velocities].get_function_gradients(y_n_solution, y_n_velocity_gradients);
    
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
        
	cell_rhs[i] += 
			  ( nu*scalar_product(grad_phi_u[i],y_n_velocity_gradients[q])  
                            + solution_velocity_values[q]*
                                  transpose(grad_phi_u[i])*y_n_velocity_values[q]        
                            + phi_u[i]*
                                  transpose(solution_velocity_gradients[q])*y_n_velocity_values[q]        
			  )
					*fe_values.JxW(q);
        
      } // loop over rows
      
    }  // loop over quadrature points q
		
    cell->get_dof_indices(local_dof_indices);     
    hanging_nodes_constraint_matrix.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, 
					   	                 g_n_system_matrix, g_n_system_rhs);      
    
  } // loop over cells	
    
  // interpolating boundary values for velocities
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(3),
                                             boundary_values,
                                             fe.component_mask(velocities));
  MatrixTools::apply_boundary_values(boundary_values,
                                       g_n_system_matrix,
                                       g_n_solution,
                                       g_n_system_rhs);
    
  timer.stop();

}


// *****************************************************************************
// Step 3b - Directly Solving for g^n using UMFPACK 
// *****************************************************************************
// Step 3 - Solving the System for g^0 in V_0
//  -> a(g^0, z) + b(z, theta) = a(y^0, z) + c(u^0, z, y^0) + c(z, u^0, y^0)
//                b(g^0, q)   = 0 
template<int dim>
void NavierStokesSolver<dim>::solve_g_n_system()
{
  printf("Solving Stokes g^n System for Least Squares Method...\n");
  Timer timer;
  timer.start ();
	
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(g_n_system_matrix);

  A_direct.vmult(g_n_solution, g_n_system_rhs);

  hanging_nodes_constraint_matrix.distribute(g_n_solution);

  timer.stop ();

}


// *******************************************************************************************
// Step 5a - Setting up the system for y_1^n - 1st Step in While Loop of Least Squares Algorithm
//           Step 5 is the 1st Step in While Loop of Least Squares Algorithm
// *******************************************************************************************
// Setting up the System for y_1^n in V_0
//  -> a(y_1^n, z) + b(z, theta) = a(w^n, z) + c(u^n, w^n, z) + c(w^n, u^n, z)
//                 b(y_1^n, q)   = 0 
template<int dim>
void NavierStokesSolver<dim>::assemble_stokes_y_1_n_system()
{
  printf("Assembling Stokes y_1^n System for Least Squares Method...\n");

  Timer timer;
  timer.start ();

  y_1_n_system_matrix = 0;    
  y_1_n_system_rhs = 0;

  ExactSolutionForcingFunction<dim>  forcing_function;		
	
  QGauss<dim>                   quadrature_formula(2);
  QGauss<dim-1>                 face_quadrature_formula(3);
	
  const int                     dofs_per_cell = fe.dofs_per_cell;
  const int                     n_q_points = quadrature_formula.size();	

  std::vector<Vector<double> >  rhs_values (n_q_points, Vector<double>(dim+1));

  std::vector<Tensor<1,dim> > 	solution_velocity_values(n_q_points);
  std::vector<Tensor< 2, dim> > 	solution_velocity_gradients(n_q_points);
  std::vector<Tensor<1,dim> > 	w_n_velocity_values(n_q_points);
  std::vector<Tensor< 2, dim> > 	w_n_velocity_gradients(n_q_points);
	
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
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_mass   = 0;
    cell_rhs = 0;

    fe_values[velocities].get_function_values(solution, solution_velocity_values);        
    fe_values[velocities].get_function_gradients(solution, solution_velocity_gradients);
    fe_values[velocities].get_function_values(w_n_solution, w_n_velocity_values);        
    fe_values[velocities].get_function_gradients(w_n_solution, w_n_velocity_gradients);
    
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
        
	cell_rhs[i] += 
			  ( nu*scalar_product(grad_phi_u[i],w_n_velocity_gradients[q])  
                            + solution_velocity_values[q]*
                                  transpose(w_n_velocity_gradients[q])*phi_u[i]        
                            + w_n_velocity_values[q]*
                                  transpose(solution_velocity_gradients[q])*phi_u[i]        
			  )
					*fe_values.JxW(q);
        
      } // loop over rows
      
    }  // loop over quadrature points q
		
    cell->get_dof_indices(local_dof_indices);     
    hanging_nodes_constraint_matrix.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, 
					   	                 y_1_n_system_matrix, y_1_n_system_rhs);      
  } // loop over cells	
    
  // interpolating boundary values for velocities
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(3),
                                             boundary_values,
                                             fe.component_mask(velocities));
  MatrixTools::apply_boundary_values(boundary_values,
                                       y_1_n_system_matrix,
                                       y_1_n_solution,
                                       y_1_n_system_rhs);
    
  timer.stop();

}


// *******************************************************************************************
// Step 5b - Solving the System for y_1^n
//           Step 5 is the 1st Step in While Loop of Least Squares Algorithm
// *******************************************************************************************
// Solving the System for y_1^n in V_0
//  -> a(y_1^n, z) + b(z, theta) = a(w^n, z) + c(u^n, w^n, z) + c(w^n, u^n, z)
//                 b(y_1^n, q)   = 0 
template<int dim>
void NavierStokesSolver<dim>::solve_y_1_n_system()
{
  printf("Solving Stokes y_1^n System for Least Squares Method...\n");
  Timer timer;
  timer.start ();
	
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(y_1_n_system_matrix);

  A_direct.vmult(y_1_n_solution, y_1_n_system_rhs);

  hanging_nodes_constraint_matrix.distribute(y_1_n_solution); 
  
  timer.stop ();

}



// *******************************************************************************************
// Step 6a - Setting up the system for y_2^n - 2nd Step in While Loop of Least Squares Algorithm
//           Step 6 is the 2nd Step in While Loop of Least Squares Algorithm
// *******************************************************************************************
// Setting up the System to Solve for y_2^n in V_0
//  -> a(y_2^n, z) + b(z, theta) = c(w^n, w^n, z) 
//                 b(y_2^n, q)   = 0 
template<int dim>
void NavierStokesSolver<dim>::assemble_stokes_y_2_n_system()
{
  printf("Assembling Stokes y_2^n System for Least Squares Method...\n");

  Timer timer;
  timer.start ();

  y_2_n_system_matrix = 0;    
  y_2_n_system_rhs = 0;

  ExactSolutionForcingFunction<dim>  forcing_function;		
	
  QGauss<dim>                   quadrature_formula(2);
  QGauss<dim-1>                 face_quadrature_formula(3);
	
  const int                     dofs_per_cell = fe.dofs_per_cell;
  const int                     n_q_points = quadrature_formula.size();	

  std::vector<Vector<double> >  rhs_values (n_q_points, Vector<double>(dim+1));

  std::vector<Tensor<1,dim> > 	w_n_velocity_values(n_q_points);
  std::vector<Tensor<2,dim> > 	w_n_velocity_gradients(n_q_points);
	
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
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_mass   = 0;
    cell_rhs = 0;

    fe_values[velocities].get_function_values(w_n_solution, w_n_velocity_values);        
    fe_values[velocities].get_function_gradients(w_n_solution, w_n_velocity_gradients);
    
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
        
	cell_rhs[i] += 
			  ( w_n_velocity_values[q]*
                                  transpose(w_n_velocity_gradients[q])*phi_u[i]        
			  )
					*fe_values.JxW(q);
        
      } // loop over rows
      
    }  // loop over quadrature points q
		
    cell->get_dof_indices(local_dof_indices);    
    hanging_nodes_constraint_matrix.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, 
					   	                 y_2_n_system_matrix, y_2_n_system_rhs);      

    
  } // loop over cells	
    
  // interpolating boundary values for velocities
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(3),
                                             boundary_values,
                                             fe.component_mask(velocities));
  MatrixTools::apply_boundary_values(boundary_values,
                                       y_2_n_system_matrix,
                                       y_2_n_solution,
                                       y_2_n_system_rhs);
    
  timer.stop();

}


// *******************************************************************************************
// Step 6b - Solving the system for y_2^n - 2nd Step in While Loop of Least Squares Algorithm
//           Step 6 is the 2nd Step in While Loop of Least Squares Algorithm
// *******************************************************************************************
// Solving the System for y_2^n in V_0
//  -> a(y_2^n, z) + b(z, theta) = c(w^n, w^n, z) 
//                 b(y_2^n, q)   = 0 
template<int dim>
void NavierStokesSolver<dim>::solve_y_2_n_system()
{
  printf("Solving Stokes y_2^n System for Least Squares Method...\n");
  Timer timer;
  timer.start ();
	
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(y_2_n_system_matrix);

  A_direct.vmult(y_2_n_solution, y_2_n_system_rhs);

  hanging_nodes_constraint_matrix.distribute(y_2_n_solution); 

  timer.stop ();

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
//
// *******************************************************************************************
// Step 7a - Calculating Derivatives of j_n - 3rd Step in While Loop of Least Squares Algorithm
//           Step 7 is the 3rd Step in While Loop of Least Squares Algorithm
// *******************************************************************************************
template<int dim>
Tensor<1,3> NavierStokesSolver<dim>::calculate_derivs_of_j_n_at_rho(double rho)
{
  printf("Calculating the Derivatives of j_n = j_n(rho) for Least Squares Method...\n");
  Timer timer;
  timer.start ();
  
  Tensor<1,3> derivs_j_n_at_rho;

  double zeroth_deriv_y_n_of_rho = 0;
  double first_deriv_y_n_of_rho = 0;
  double second_deriv_y_n_of_rho = 0;

  double a_of_y_n_y_n = 0;
  double a_of_y_n_y_1_n = 0;
  double a_of_y_n_y_2_n = 0;   
  double a_of_y_1_n_y_1_n = 0;
  double a_of_y_1_n_y_2_n = 0;
  double a_of_y_2_n_y_2_n = 0;

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
  
    fe_values.reinit(cell);
    
    fe_values[velocities].get_function_gradients(y_n_solution, y_n_velocity_gradients);        
    fe_values[velocities].get_function_gradients(y_1_n_solution, y_1_n_velocity_gradients);        
    fe_values[velocities].get_function_gradients(y_2_n_solution, y_2_n_velocity_gradients);        

    //calculate cell contributions through quadrature
    for (int q = 0; q < n_q_points; q++)
    {
      a_of_y_n_y_n += (nu*scalar_product(y_n_velocity_gradients[q],
                                         y_n_velocity_gradients[q])) * fe_values.JxW(q);
      a_of_y_n_y_1_n += (nu*scalar_product(y_n_velocity_gradients[q],
                                           y_1_n_velocity_gradients[q])) * fe_values.JxW(q);
      a_of_y_n_y_2_n += (nu*scalar_product(y_n_velocity_gradients[q],
                                           y_2_n_velocity_gradients[q])) * fe_values.JxW(q);
      a_of_y_1_n_y_1_n += (nu*scalar_product(y_1_n_velocity_gradients[q],
                                             y_1_n_velocity_gradients[q])) * fe_values.JxW(q);
      a_of_y_1_n_y_2_n += (nu*scalar_product(y_1_n_velocity_gradients[q],
                                             y_2_n_velocity_gradients[q])) * fe_values.JxW(q);
      a_of_y_2_n_y_2_n += (nu*scalar_product(y_2_n_velocity_gradients[q],
                                             y_2_n_velocity_gradients[q])) * fe_values.JxW(q);

    } // quadrature loop

    
  } // cell loop


  double rho_squared = rho * rho;
  double rho_cubed = rho_squared * rho;
  double rho_fourth = rho_cubed * rho;
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


// *****************************************************************************
// Step 3d, 9d - Calculating a(g_n,g_n) from Solution of g^n System 
// *****************************************************************************
template<int dim>
double NavierStokesSolver<dim>::calculate_a_of_g_n_g_n()
{
  printf("Calculating a(g_n,g_n) for relative error for Least Squares Method...\n");
  Timer timer;
  timer.start ();
  
  double a_of_g_n_g_n = 0;

  QGauss<dim>                 quadrature_formula(4);

  const int                   dofs_per_cell = fe.dofs_per_cell;
  const int                   n_q_points = quadrature_formula.size();	

  std::vector<Tensor<2,dim> >        g_n_velocity_gradients(n_q_points);

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
  
    fe_values.reinit(cell);
    
    fe_values[velocities].get_function_gradients(g_n_solution, g_n_velocity_gradients);        

    //calculate cell contributions through quadrature
    for (int q = 0; q < n_q_points; q++)
    {
      a_of_g_n_g_n += (nu*scalar_product(g_n_velocity_gradients[q],
                                         g_n_velocity_gradients[q])) * fe_values.JxW(q);

    } // quadrature loop

    
  } // cell loop

  return a_of_g_n_g_n;

}




// *****************************************************************************
// Step 2d - Calculating a(y_n,y_n) from Solution of y^n System 
// *****************************************************************************
template<int dim>
double NavierStokesSolver<dim>::calculate_a_of_y_n_y_n()
{
  printf("Calculating a(y_n,y_n) for relative error for Least Squares Method...\n");
  Timer timer;
  timer.start ();
  
  double a_of_y_n_y_n = 0;

  QGauss<dim>                 quadrature_formula(4);

  const int                   dofs_per_cell = fe.dofs_per_cell;
  const int                   n_q_points = quadrature_formula.size();	

  std::vector<Tensor<2,dim> >        y_n_velocity_gradients(n_q_points);

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
    fe_values.reinit(cell);
    
    fe_values[velocities].get_function_gradients(y_n_solution, y_n_velocity_gradients);        

    //calculate cell contributions through quadrature
    for (int q = 0; q < n_q_points; q++)
    {
      a_of_y_n_y_n += (nu*scalar_product(y_n_velocity_gradients[q],
                                         y_n_velocity_gradients[q])) * fe_values.JxW(q);
    } // quadrature loop
    
  } // cell loop

  return a_of_y_n_y_n;

}



int main ()
{

  const int dim = 2;
  unsigned int cells_per_coordinate_direction = 64;
  NavierStokesSolver<dim> navier_stokes_problem(cells_per_coordinate_direction);

  navier_stokes_problem.run_least_squares_loop();
}





