#include "bits/stdc++.h"

#define Null               -1 
#define Inf_Cost           1000000000
#define Magnify_Rate       10000
#define Max_Inst_Num       10000 
#define Max_City_Num       10000
#define Max_Candidate_Num  1000 
#define Max_Depth  		   10

#define Random_Seed  489663920

using SqMat = std::vector<std::vector<double>>;

//Hyper parameters 
static double Alpha=1;       //used in estimating the potential of each edge
static double Beta=10;       //used in back propagation
static double Param_H=10;   //used to control the number of sampling actions
static double Param_T=0.10;  	 //used to control the termination condition

class MCTS {
public:
	using state_type = std::list<int>;
	using action_type = std::vector<int>;

	MCTS(SqMat& heatmap, SqMat& dist_mat, int max_allowed_time);

	// void enumerate_within_small_nbhd() {
	// 	while (!time_out()) {
	// 		optimize(2);
	// 	}
	// }

	void optimize() {
		optimize(10);
	}
	state_type get_random_state();
	double get_distance() {
		return get_distance(state);
	}
	double get_distance(const std::list<int>& state) {
		double dist = 0;
		assert(state.size() == num_of_nodes);
		auto cur = state.begin();
		while (std::next(cur) != state.end()) {
			assert(*cur != *std::next(cur));
			dist += dist_mat[*cur][*std::next(cur)];
			++cur;
		}
		return dist;
	}

	bool time_out() {
		auto time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(time - start_time);
		return (duration.count() >= max_allowed_time);
	}

private:
	state_type state;
	int num_of_nodes;
	SqMat heatmap;
	SqMat dist_mat;
	SqMat W;
	SqMat Q;
	int M;
	decltype(std::chrono::high_resolution_clock::now()) start_time;
	int max_allowed_time;
	
	void optimize(int k_max);

	double get_improvement(action_type& action) {
		int k = action.size() / 2;
		bool complete_action = action.size() % 2;
		if (!complete_action) {
			action.push_back(action[0]);
		} else {
			assert(action[0] == action[action.size() - 1]);
		}
		double improvement = 0;
		for (int i = 0; i < k; i += 2) {
			improvement += dist_mat[i + 1][i + 2] - dist_mat[i][i + 1];
		}
		if (!complete_action) {
			action.pop_back();
		}
		return improvement;
	}
};

double get_random_number() {
	// Create a random number generator engine
	std::mt19937 rng(std::random_device{}());

	// Create a uniform real distribution in the range [0, 1)
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	// Generate a random number
	return dist(rng);
}

template<typename _key, typename _val>
_key sample_from_map(const std::unordered_map<_key, _val>& map) {
	double sum = std::accumulate(map.begin(), map.end(), 0., [](auto a, auto b) { return a + b.second; });
	double sample = get_random_number() * sum;
	for (auto [j, val]: map) {
		if (sample < val) {
			return j;
		}
		sample -= val;
	}
	assert(false);
}