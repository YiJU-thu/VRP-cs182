#include "MCTS.h"
#include <bits/stdc++.h>

using distance_t = int;
// using SqMat = std::vector<std::vector<double>>;

static double get_distance(SqMat& dist_mat, std::vector<int> path);

static inline auto SqMat_n(int n) {
  return SqMat(n, SqMat::value_type(n));
}

// For TSP20-50-100-200-500-1000 instances
void Solve_One_Instance(int Inst_Index, std::fstream& fs, MCTS& problem)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  problem.optimize();

// 	double Stored_Solution_Double_Distance=Get_Stored_Solution_Double_Distance(Inst_Index);
// 	double Current_Solution_Double_Distance=Get_Current_Solution_Double_Distance();

// 	if(Stored_Solution_Double_Distance/Magnify_Rate-Current_Solution_Double_Distance/Magnify_Rate > 0.000001)
// 		Beat_Best_Known_Times++;
// 	else if(Current_Solution_Double_Distance/Magnify_Rate-Stored_Solution_Double_Distance/Magnify_Rate > 0.000001)
// 		Miss_Best_Known_Times++;
// 	else
// 		Match_Best_Known_Times++;

// 	Sum_Opt_Distance+=Stored_Solution_Double_Distance/Magnify_Rate;
// 	Sum_My_Distance+=Current_Solution_Double_Distance/Magnify_Rate;
// 	Sum_Gap += (Current_Solution_Double_Distance-Stored_Solution_Double_Distance)/Stored_Solution_Double_Distance;

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

// 	printf("\nInst_Index:%d Concorde Distance:%f, MCTS Distance:%f Improve:%f Time:%.2f Seconds\n", Inst_Index+1, Stored_Solution_Double_Distance/Magnify_Rate,
// 			Current_Solution_Double_Distance/Magnify_Rate, Stored_Solution_Double_Distance/Magnify_Rate-Current_Solution_Double_Distance/Magnify_Rate, ((double)clock()-Current_Instance_Begin_Time)/CLOCKS_PER_SEC);

// 	FILE *fp;
// 	fp=fopen(Statistics_File_Name, "a+");
// 	fprintf(fp,"\nInst_Index:%d \t City_Num:%d \t Concorde:%f \t MCTS:%f Improve:%f \t Time:%.2f Seconds\n",Inst_Index+1, Virtual_City_Num, Stored_Solution_Double_Distance/Magnify_Rate,
// 			Current_Solution_Double_Distance/Magnify_Rate, Stored_Solution_Double_Distance/Magnify_Rate-Current_Solution_Double_Distance/Magnify_Rate, ((double)clock()-Current_Instance_Begin_Time)/CLOCKS_PER_SEC);

// 	fprintf(fp,"Solution: ");
// 	int Cur_City=Start_City;
// 	do
// 	{
// 		fprintf(fp,"%d ",Cur_City+1);
// 		Cur_City=All_Node[Cur_City].Next_City;
// 	}while(Cur_City != Null && Cur_City != Start_City);

// 	fprintf(fp,"\n");
// 	fclose(fp);

// 	Release_Memory(Virtual_City_Num);
}

bool Solve_Instances_In_Batch(int thread_num, int num_of_nodes, int batch_start_id, int batch_end_id, std::vector<SqMat>& heatmaps, std::vector<SqMat>& dist_mats)
{
  using std::to_string;
  std::string output_filename = "./results/tsp" + to_string(num_of_nodes) + "/result_" 
                  + to_string(thread_num) + ".txt";

  std::fstream fs(output_filename, std::ios::out);
  if (fs.fail()) {
    std::cerr << "Cannot open output file " << output_filename << std::endl;
  }

  fs << "Number_of_Instances_In_Current_Batch: " << batch_end_id - batch_start_id << '\n';

  for (int i = batch_start_id; i < batch_end_id; i++) {
    SqMat& heatmap = heatmaps[i];
    SqMat& dist_mat = dist_mats[i];
    MCTS problem(heatmap, dist_mat, 0.1 * num_of_nodes);
    Solve_One_Instance(i, fs, problem);
  }

  fs.close();

    return true;
}

int main(int argc, char ** argv)
{
  using std::stoi, std::cout;
  using std::vector, std::string;
  double Overall_Begin_Time=(double)clock();

  srand(Random_Seed);

  // puts(argv[0]);
  std::cout << argv[0] << std::endl;

  assert(argc == 4);

  int num_threads = stoi(argv[1]);
  string Input_File_Name = argv[2];
  int num_of_nodes = stoi(argv[3]);

  // read in num_of_nodes and num_of_instances

  int num_of_instances = 10000;
  int batch_size = std::ceil(static_cast<double>(num_of_instances) / num_threads);

  // read in distance matrix and heatmap
  string heatmap_filename = "data/tsp" + std::to_string(num_of_nodes) + "/heatmap.txt";
  string dist_mat_filename = "data/tsp" + std::to_string(num_of_nodes) + "/dist_mat.txt";

  std::fstream heatmap_fs(heatmap_filename, std::ios::in);
  std::fstream dist_mat_fs(dist_mat_filename, std::ios::in);

  if (heatmap_fs.fail()) {
    std::cerr << heatmap_filename << " does not exist!" << std::endl;
    exit(1);
  }
  if (dist_mat_fs.fail()) {
    std::cerr << dist_mat_filename << " does not exist!" << std::endl;
    exit(1);
  }

  vector<SqMat> heatmaps;
  vector<SqMat> dist_mats;
  for (int i = 0; i < num_of_instances; i++) {
    SqMat heatmap = SqMat_n(num_of_nodes);
    SqMat dist_mat = SqMat_n(num_of_nodes);
    for (int j = 0; j < num_of_nodes; j++) {
      for (int k = 0; k < num_of_nodes; k++) {
        heatmap_fs >> heatmap[j][k];
        dist_mat_fs >> dist_mat[j][k];
      }
    }
    heatmaps.emplace_back(std::move(heatmap));
    dist_mats.emplace_back(std::move(dist_mat));
  }

  heatmap_fs.close();
  dist_mat_fs.close();

  vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    int start = i * batch_size;
    int end = std::min(start + batch_size, num_of_instances);
    auto f = [&](int i){ Solve_Instances_In_Batch(i, num_of_nodes, start, end, heatmaps, dist_mats); };
    threads.emplace_back(f, i);
  }

  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  

    // std::fstream fs(Statistics_File_Name, std::ios::out | std::ios::app);

    // fs << "\n\nAvg_Concorde_Distance: " << (Sum_Opt_Distance / Test_Inst_Num)
    //           << " Avg_MCTS_Distance: " << (Sum_My_Distance / Test_Inst_Num)
    //           << " Avg_Gap: " << (Sum_Gap / Test_Inst_Num)
    //           << " Total_Time: " << ((double)clock() - Overall_Begin_Time) / CLOCKS_PER_SEC
    //           << " Seconds \n Beat_Best_Known_Times: " << Beat_Best_Known_Times
    //           << " Match_Best_Known_Times: " << Match_Best_Known_Times
    //           << " Miss_Best_Known_Times: " << Miss_Best_Known_Times << " \n";

  // fs.close();

  // FILE *fp;
  // fp=fopen(Statistics_File_Name, "a+");
  // fprintf(fp,"\n\nAvg_Concorde_Distance: %f Avg_MCTS_Distance: %f Avg_Gap: %f Total_Time: %.2f Seconds \n Beat_Best_Known_Times: %d Match_Best_Known_Times: %d Miss_Best_Known_Times: %d \n",
  // 		Sum_Opt_Distance/Test_Inst_Num,Sum_My_Distance/Test_Inst_Num, Sum_Gap/Test_Inst_Num, ((double)clock()-Overall_Begin_Time)/CLOCKS_PER_SEC, Beat_Best_Known_Times, Match_Best_Known_Times, Miss_Best_Known_Times);
  // fclose(fp);

  // cout << "\n\nAvg_Concorde_Distance: " << (Sum_Opt_Distance / Test_Inst_Num)
    //      << " Avg_MCTS_Distance: " << (Sum_My_Distance / Test_Inst_Num)
    //      << " Avg_Gap: " << (Sum_Gap / Test_Inst_Num);
    // cout << " Total_Time: " << std::setprecision(2) 
  // 		<< (static_cast<double>(clock()) - Overall_Begin_Time) / CLOCKS_PER_SEC 
  // 		<< " Seconds \n"
    //      << " Beat_Best_Known_Times: " << Beat_Best_Known_Times
    //      << " Match_Best_Known_Times: " << Match_Best_Known_Times
    //      << " Miss_Best_Known_Times: " << Miss_Best_Known_Times << " \n";

  // printf("\n\nAvg_Concorde_Distance: %f Avg_MCTS_Distance: %f Avg_Gap: %f Total_Time: %.2f Seconds \n Beat_Best_Known_Times: %d Match_Best_Known_Times: %d Miss_Best_Known_Times: %d \n",
  // 		Sum_Opt_Distance/Test_Inst_Num,Sum_My_Distance/Test_Inst_Num, Sum_Gap/Test_Inst_Num, ((double)clock()-Overall_Begin_Time)/CLOCKS_PER_SEC, Beat_Best_Known_Times, Match_Best_Known_Times, Miss_Best_Known_Times);
  // getchar();

  return 0;
}

MCTS::MCTS(SqMat& heatmap, SqMat& dist_mat, int max_allowed_time) : heatmap(heatmap), dist_mat(dist_mat), max_allowed_time(max_allowed_time), M(0) {
  // TODO: add sanity checks to heatmap and dist_mat
  num_of_nodes = heatmap.size();
  W = SqMat_n(num_of_nodes);
  Q = SqMat_n(num_of_nodes);
  M = 0;
  state = get_random_state();
  start_time = std::chrono::high_resolution_clock::now();

  // TODO: maybe enforce W to be symmetric?

  // for (int i = 0; i < num_of_nodes; i++) {
  // 	for (int j = i + 1; j < num_of_nodes; j++) {
  // 		W[j][i] = W[i][j] = 50 * (heatmap[i][j] + heatmap[j][i]);
  // 	}
  // }

  for (int i = 0; i < num_of_nodes; i++) {
    for (int j = 0; j < num_of_nodes; j++) {
      W[i][j] = 100 * heatmap[i][j];
    }
  }

}

void MCTS::optimize(int k_max) {
  while (!time_out()) {
    assert(state.size() == num_of_nodes);
    auto new_state = state;
    
    // simulation
    int a1 = std::rand() % num_of_nodes;
    auto it = std::find(new_state.begin(), new_state.end(), a1);
    assert(it != new_state.end());
    int ai = a1;
    int bi;
    action_type action;
    double improvement = Inf_Cost;
    for (int i = 0; i < k_max; i++) {
      if (it != new_state.begin()) {
        new_state.splice(new_state.end(), new_state, new_state.begin(), it);
      }
      bi = *new_state.rbegin();
      if ((improvement = get_improvement(action)) < 0) {
        action.push_back(a1);
        break;
      }
      action.push_back(ai);
      action.push_back(bi);

      // get the set X and the coresponding Z-values
      std::unordered_map<int, double> map;
          auto& w = W[bi];
          auto& q = Q[bi];
          double omega = std::accumulate(w.begin(), w.end(), 0);
          auto Z = [&](int j) { return w[j] / omega + Alpha * std::sqrt(std::log(M + 1) / (q[j] + 1)); };
          for (int j = 0; j < num_of_nodes; j++) {
              if (bi != j && bi != a1 && w[j] >= 1) {
                  double Z_j = Z(j);
                  map.emplace(j, Z(j));
              }
          }

      ai = sample_from_map(map);
    }

    // selection and back-prop
    if (improvement < 0) {
      M++;
      for (int i = 1; i < action.size(); i += 2) {
        Q[action[i]][action[i + 1]]++;
        W[action[i]][action[i + 1]] += Beta * (std::exp(improvement / get_distance(state)) - 1);
      }
      state = new_state;
    } else {
      state = get_random_state();
    }
  }
}

MCTS::state_type MCTS::get_random_state() {
  int s1 = std::rand() % num_of_nodes;
  state_type state{s1};
  std::unordered_map<int, double> map;
  for (int i = 0; i < num_of_nodes; i++) {
    map.emplace(i, heatmap[s1][i]);
  }
  map.erase(s1);
  for (int i = 1; i < num_of_nodes; i++) {
    assert(map.size() == num_of_nodes - i);
    int si = sample_from_map(map);
    state.push_back(si);
    map.erase(si);
    for (auto& [j, val]: map) {
      val = heatmap[si][j];
    }
  }
  return state;
}