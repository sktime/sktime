#include "seql.h"

#include <iostream>
#include <fstream>
#include <set>

void read_multi_data(string path, vector<vector<string>> &sequences,vector<double> &labels){
	ifstream in(path);


	std::string str;
	int last_config = -1;

	//bool print = true;

	while (std::getline(in, str)) {
		// extract index
		int first_space = str.find_first_of(" ");
		int cr_config = stoi(str.substr(0,first_space));
		str = str.substr(first_space+1);
		// extract class
		int second_space = str.find_first_of(" ");
		double label;
		if (second_space == string::npos){ // empty string
			label = stof(str);
			str = "";
		} else {
			label = stof(str.substr(0,second_space));
			// only populate labels vector once as it repeats
			str = str.substr(second_space+1);
		}
		//std::cout << cr_config << ":" << label << std::endl;
		// sequence

		if (cr_config == 0){
			labels.push_back(label);
		}

		if (cr_config != last_config){
			sequences.push_back(vector<string>());
		}
		//if (print){
		//cout << str << endl;
		//print = false;
		//}
		//if (str.length() <= 2){
		//	cout << "Warning:|" << str << "|:Warning" << endl;
		//}
		sequences[cr_config].push_back(str);

		last_config = cr_config;

	}

	in.close();
}


void write_matrix_to_files(vector<vector<int>>& mt, vector<double>& lb, string path){
	std::ofstream file (path);

	for (int i = 0; i < mt.size(); i++){
		file << lb[i];
		for (int v: mt[i]){
			file << " " << v;
		}
		file << "\n";
	}

	file.close();
}

void write_matrix_to_files(vector<vector<int>>& mt, string path){
	std::ofstream file (path);

	for (int i = 0; i < mt.size(); i++){
		for (int j = 0; j < mt[i].size(); j++){
			file << mt[i][j];
			if (j < mt[i].size() - 1){
				file << " ";
			}
		}
		if (i < mt.size() - 1){
			file << "\n";
		}
	}

	file.close();
}

void write_vector_to_file(vector<double>& lb, string path){
	std::ofstream file (path);
	for (int i = 0; i < lb.size(); i++){
		file << lb[i];
		if (i < lb.size() - 1){
			file << " ";
		}
	}
	file.close();
}

void multiclass_ensemble(vector<vector<string>> &train_sequences, vector<double> &train_labels,
		vector<vector<string>> &test_sequences, vector<double> &test_labels,
		string fs_output, bool fs){

	double total_learn_time = 0;
	double max_learn_time = 0;
	double total_classify_time = 0;
	double max_classify_time = 0;
	double total_vectorspace_time = 0;

	//string output;

	//vector<vector<string>> train_sequences;
	//vector<double> train_labels;

	//vector<vector<string>> test_sequences;
	//vector<double> test_labels;

	set<double> unique_labels;

	std::ofstream* ft_file;

	//read_multi_data(train_data,train_sequences,train_labels);
	//read_multi_data(test_data,test_sequences,test_labels);

	if (fs){
		ft_file = new std::ofstream(fs_output+"/features");
		write_vector_to_file(train_labels,fs_output+"/train.y");
		write_vector_to_file(test_labels,fs_output+"/test.y");
	}

	for (double lb: train_labels){
		unique_labels.insert(lb);
	}
	cout << "Number of classes: " << unique_labels.size() << endl;

	// number of representation
	int cfg_count = train_sequences.size();


	vector<double> final_scores;
	vector<double> predictions;
	final_scores.resize(test_labels.size());
	std::fill(final_scores.begin(), final_scores.end(), -numeric_limits<double>::max());
	predictions.resize(test_labels.size());



	for (double current_label: unique_labels){
		vector<vector<int>> train_fs;
		vector<vector<int>> test_fs;
		vector<double> bin_train;
		vector<double> bin_test;
		vector<double> lb_scores;
		lb_scores.resize(test_labels.size());
		std::fill(lb_scores.begin(), lb_scores.end(), 0.0);
		// convert to binary labels

		for (double lb: train_labels){
			if (lb == current_label){
				bin_train.push_back(1);
			} else {
				bin_train.push_back(-1);
			}
		}
		for (double lb: test_labels){
			if (lb == current_label){
				bin_test.push_back(1);
			} else {
				bin_test.push_back(-1);
			}
		}
		// run seql
		for (int i = 0; i < cfg_count;++i){
			SEQL seql;
			seql.run_sax_seql(train_sequences[i],bin_train,test_sequences[i],bin_test);
			// time measurement
			total_learn_time += seql.get_last_learn_time();
			total_classify_time += seql.get_last_classification_time();
			max_learn_time = (max_learn_time < seql.get_last_learn_time()) ? seql.get_last_learn_time() : max_learn_time;
			max_classify_time = (max_classify_time < seql.get_last_classification_time()) ? seql.get_last_classification_time() : max_classify_time;

			for (int j = 0; j < test_labels.size(); ++j){
				lb_scores[j] += (*seql.get_test_scores())[j];
			}


			// creating feature vector space
			if (fs){
				// write down features
				for( const auto& fv_pair : seql.model )
				{
					if (fv_pair.second > 0){
						(*ft_file) << i << "," << fv_pair.second << "," << fv_pair.first << endl;
					}
				}
				clock_t vs_starttime = clock();
				seql.to_positive_feature_space(train_sequences[i],train_fs);
				seql.to_positive_feature_space(test_sequences[i],test_fs);
				total_vectorspace_time += double(clock() - vs_starttime) / CLOCKS_PER_SEC;
			}
		}
		if (fs){
			write_matrix_to_files(train_fs,fs_output+"/train.x."+to_string((int)current_label));
			write_matrix_to_files(test_fs,fs_output+"/test.x."+to_string((int)current_label));
		}
		// prediction ~ max prediction score
		for (int i = 0; i < test_labels.size(); ++i){
			if (lb_scores[i] > final_scores[i]){
				final_scores[i] = lb_scores[i];
				predictions[i] = current_label;
			}
		}


	}

	if (fs){
		ft_file->close();
	}

	int correct = 0;
	for (int i = 0; i < test_labels.size(); ++i){
		if (test_labels[i] == predictions[i]){
			correct++;
		}
	}
	cout << "Elapsed Time (TotalLearn,MaxLearn,TotalTest,MaxTest,VectorSpace):"
			<< total_learn_time << ","
			<< max_learn_time << ","
			<< total_classify_time << ","
			<< max_classify_time << ","
			<< total_vectorspace_time
			<< endl;
	cout << "(Ensemble) SEQL Accuracy: " << correct*1.0/test_labels.size() << endl;
	//return 1.0 - correct*1.0/test_labels.size();

}

//void bin_ensemble(string train_data, string test_data, string fs_output, bool fs){
void bin_ensemble(vector<vector<string>> &train_sequences, vector<double> &train_labels,
		vector<vector<string>> &test_sequences, vector<double> &test_labels,
		string fs_output, bool fs){
	double total_learn_time = 0;
	double max_learn_time = 0;
	double total_classify_time = 0;
	double max_classify_time = 0;
	double total_vectorspace_time = 0;

	std::ofstream* ft_file;

	// convert labels to 1 and -1
	double posy = train_labels[0];
	for (int i = 0; i < train_labels.size(); ++i){
		if (train_labels[i] == posy){
			train_labels[i] = 1;
		} else {
			train_labels[i] = -1;
		}
	}
	for (int i = 0; i < test_labels.size(); ++i){
		if (test_labels[i] == posy){
			test_labels[i] = 1;
		} else {
			test_labels[i] = -1;
		}
	}

	if (fs){
		ft_file = new std::ofstream(fs_output+"/features");
		write_vector_to_file(train_labels,fs_output+"/train.y");
		write_vector_to_file(test_labels,fs_output+"/test.y");
	}

	int cfg_count = train_sequences.size();


	vector<double> total_scores;
	total_scores.resize(test_labels.size());
	std::fill(total_scores.begin(), total_scores.end(), 0.0);

	vector<vector<int>> train_fs;
	vector<vector<int>> test_fs;

	for (int i = 0; i < cfg_count;++i){
		SEQL seql;
		double error = seql.run_sax_seql(train_sequences[i],train_labels,test_sequences[i],test_labels);

		// time measurement
		total_learn_time += seql.get_last_learn_time();
		total_classify_time += seql.get_last_classification_time();
		max_learn_time = (max_learn_time < seql.get_last_learn_time()) ? seql.get_last_learn_time() : max_learn_time;
		max_classify_time = (max_classify_time < seql.get_last_classification_time()) ? seql.get_last_classification_time() : max_classify_time;



		//cout << "SFA-SEQL:" << error << endl;
		//seql.print_model(20);
		for (int j = 0; j < test_labels.size(); ++j){
			total_scores[j] += (*seql.get_test_scores())[j];
		}
		if (fs){
			// write down features
			for( const auto& fv_pair : seql.model )
			{
				(*ft_file) << i << "," << fv_pair.second << "," << fv_pair.first << endl;
			}
			// to feature vectors
			clock_t vs_starttime = clock();
			seql.to_feature_space(train_sequences[i],train_fs);
			seql.to_feature_space(test_sequences[i],test_fs);
			total_vectorspace_time += double(clock() - vs_starttime) / CLOCKS_PER_SEC;

		}

	}
	if (fs) {
		ft_file->close();
		write_matrix_to_files(train_fs,fs_output+"/train.x");
		write_matrix_to_files(test_fs,fs_output+"/test.x");
	}

	int correct = 0;
	int tp = 0,fp = 0,tn = 0,fn = 0;

	for (int i = 0; i < test_labels.size();i++){
    //cout << total_scores[i] << endl;
		if (total_scores[i] > 0){
			if (test_labels[i] > 0){
				correct++;
				tp++;
			} else {
				fp++;
			}
		} else {
			if (test_labels[i] < 0){
				correct++;
				tn++;
			} else {
				fn++;
			}
		}
	}




	cout << "Elapsed Time (TotalLearn,MaxLearn,TotalTest,MaxTest,VectorSpace):"
			<< total_learn_time << ","
			<< max_learn_time << ","
			<< total_classify_time << ","
			<< max_classify_time << ","
			<< total_vectorspace_time
			<< endl;

	cout << "(Ensemble) SEQL Accuracy: " << 1.0*correct/test_labels.size() << endl;
	cout << "TP/FP/TN/FN: " << tp << "/" << fp << "/" << tn << "/" << fn << endl;
}

// to automatically call bin_ensemble or multi_ensemble depends on the number of classes in the dataset
void ensemble_seql(string train_data, string test_data, string fs_output, bool fs){
	vector<vector<string>> train_sequences;
	vector<double> train_labels;

	vector<vector<string>> test_sequences;
	vector<double> test_labels;


	read_multi_data(train_data,train_sequences,train_labels);
	read_multi_data(test_data,test_sequences,test_labels);


	cout << "Number of representations:" << train_sequences.size() << endl;

	double first_label = train_labels[0];
	double second_label = first_label;

	bool isBinary = true;
	for (double l : train_labels){
		if (l != first_label && l!= second_label){
			if (first_label == second_label){
				second_label = l;
			} else { // there are more than 2 classes in the data
				isBinary = false;
				break;
			}
		}
	}

	if (isBinary){
		bin_ensemble(train_sequences,train_labels,test_sequences,test_labels,fs_output,fs);
	} else {
		multiclass_ensemble(train_sequences,train_labels,test_sequences,test_labels,fs_output,fs);
	}
}

void test_function(){
  cout << "Helloooooo !!" << endl;
}
