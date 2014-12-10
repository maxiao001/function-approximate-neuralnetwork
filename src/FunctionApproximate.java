import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class FunctionApproximate {
	
	
	double []input;
	int iter_size = 1000000;
	int loss_itr_size = 1000;
	int test_size = 10000;
	NeuralNetwork network;
	double []last_layer_grad;
	double learn_rate = 0.01;
	

	Random r = new Random(10);
	public void init(){
		network = new NeuralNetwork();
		List<Integer> layer_units_list = new ArrayList<Integer>();
		layer_units_list.add(2);
		layer_units_list.add(100);
		//layer_units_list.add(40);
		layer_units_list.add(1);
		List<ActivationFunc> layer_activation_fun_list = new ArrayList<ActivationFunc>();
		layer_activation_fun_list.add(new LinearFunc());
		layer_activation_fun_list.add(new SigmoidFunc());
		//layer_activation_fun_list.add(new SigmoidFunc());
		layer_activation_fun_list.add(new LinearFunc());
		network.init(layer_units_list, layer_activation_fun_list);
		input = new double[2];
		last_layer_grad = new double[1];
		
	}
	private void random_input() {
		
		for (int i = 0; i < input.length; i++){
			input[i] = (double)r.nextInt(1000)/100-5;
		}
	}
	public double function_vector_product(){
		double result = 0;
		for(int i = 0;i < input.length;i ++){
			result += input[i]*input[i];
		}
		return result;
	}
	public double function_vector_sum(){
		double result = 0;
		for(int i = 0;i < input.length;i ++){
			result += input[i];//+//Math.cos(input[i])+Math.sin(15*input[i])+input[i]*input[i];
		}
		return result;
	}
	public double function_vector_part_cosine(){
		double inner_product = 0;
		double length1 = 0;
		double length2 = 0;
		for(int i = 0;i < input.length/2;i ++){
			inner_product += input[i]*input[i+input.length/2];
			length1 += input[i]*input[i];
			length2 += input[i+input.length/2]*input[i+input.length/2];
		}
		return inner_product/(Math.sqrt(length1)*Math.sqrt(length2));
	}
	public double function_vector_part_x_y(){
		return input[0]*input[1];
	}
	//square loss
	private void caculate_last_layer_grad() {
		double real_value = main_function_value();
		last_layer_grad[0] = network.a_buf[network.total_unit_num-1]-real_value;
	}
	private double caculate_loss(){
		double real_value = main_function_value();
		double dis = network.a_buf[network.total_unit_num-1]-real_value;
		return 0.5*dis*dis;
	}
	private double main_function_value() {
		return function_vector_part_x_y();
	}
	public void train(){
		
		double loss = 0.0;
		for(int i = 0;i < iter_size;i ++){
			if(i % loss_itr_size == 0 && i > 0){
				System.out.println("average train loss after "+loss_itr_size+" pairs:"+loss/loss_itr_size);
				loss = 0;
			}
			random_input();
			network.zero_grad();
			network.forward_propagation(input);
			caculate_last_layer_grad();
			loss += caculate_loss();
			network.back_propagation(last_layer_grad);
			//this.do_gradient_checking();
			network.update_par(learn_rate);
			
		}
	}
	public void test(){
		double loss = 0;
		for(int i = 0;i < test_size;i++){
			random_input();
			network.forward_propagation(input);
			loss += caculate_loss();
		}
		System.out.println("average test loss:"+loss/test_size);
	}
	public void do_gradient_checking() {
		double delta = 0.000001;
		double add_value = 0;
		double sub_value = 0;
		for(int i = 0;i < network.par.length;i ++){
			double real_par = network.par[i];
			network.par[i] = real_par+delta;
			network.forward_propagation(input);
			add_value = caculate_loss();
			
			network.par[i] = real_par-delta;
			network.forward_propagation(input);
			sub_value = caculate_loss();
			//if(Math.abs((add_value-sub_value)/(2*delta)-network.grad[i]) > 0.0000001){
				System.out.println("index:"+i+" real grad:"+network.grad[i]+" checking_grad:"+(add_value-sub_value)/(2*delta));
			//}
		}
	}



}
