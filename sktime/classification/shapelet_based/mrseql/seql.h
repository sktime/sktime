/*
 * seql.h
 *
 *  Created on: 4 Dec 2017
 *      Author: thachln
 */

#ifndef SEQL_H_
#define SEQL_H_

#include "seql_learn.h"

#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <map>
#include <iterator>



using namespace std;

class SEQL {

private:

	int verbose = 1;

	bool token_type = 1;
	vector<double> test_scores;

	double lastrun_learn_time;
	double lastrun_classification_time;

public:
	map<string,double> model;
	// return size of data by counting number of lines
	void learn(vector<string>& sequences, vector<double>& _labels ){
		model.clear();

		SeqLearner seql_learner;
		unsigned int objective = 0;
		unsigned int maxpat = 0xffffffff;
		unsigned int minpat = 1;
		unsigned int maxitr = 5000;
		unsigned int minsup = 1;

		// Max # of total wildcards allowed in a feature.
		unsigned int maxgap = 0;
		// Max # of consec wildcards allowed in a feature.
		unsigned int maxcongap = 0;
		// Word or character token type. By default char token.
		//bool token_type = 0;
		// BFS vs DFS traversal. By default BFS.
		bool traversal_strategy = 0;

		// The C regularizer parameter in regularized loss formulation. It constraints the weights of features.
		// C = 0 no constraints (standard SLR), the larger the C, the more the weights are shrinked towards each other (using L2) or towards 0 (using L1)
		double C = 1;
		// The alpha parameter decides weight on L1 vs L2 regularizer: alpha * L1 + (1 - alpha) * L2. By default we use an L2 regularizer.
		double alpha = 0.2;

		double convergence_threshold = 0.005; //default: 0.005
		int verbosity = 0;

		seql_learner.external_read(sequences,_labels);

		seql_learner.run_internal (model, objective, maxpat, minpat, maxitr,
				minsup, maxgap, maxcongap, token_type, traversal_strategy, convergence_threshold, C, alpha, verbosity);
	}

	double brute_classify(string sequence, double threshold = 0.0){
		std::map<std::string, double>::iterator it = model.begin();
		double score = 0;
		while(it != model.end())
		{
			//if (sequence.find(it->first) != std::string::npos){
			if (contain_pattern(sequence,it->first)){
				score += it->second;
			}

			it++;
		}

		return score;
	}

	bool contain_pattern(string sequence, string pattern){
		int si = 0;
		int pi = 0;
		int diff = int(sequence.length()) - int(pattern.length());
		while(si <= diff){
			int match_count = 0;
			while ((match_count < pattern.length()) &&
					(sequence.at(si+match_count) == pattern.at(pi+match_count) || pattern.at(pi+match_count) == '*'))
			{
				match_count++;
			}
			if (match_count == pattern.length()){
				return true;
			}
			si++;
		}
		return false;
	}


	double tune_threshold(vector<string> &sequences, vector<double> &labels){
		unsigned int verbose = 0;
		double threshold = 0; // By default zero threshold = zero bias.
		vector<pair<double,double>> scores;

		// Predicted and true scores for all docs.
		//vector<pair<double, int> > scores;


		unsigned int correct = 0;
		// Total number of true positives.
		unsigned int num_positives = 0;




		for (unsigned int item = 0; item < sequences.size();++item){
			//double predicted_score = seql.classify_with_mytrie(sequences[item].c_str(), max_distance);
			double predicted_score = brute_classify(sequences[item].c_str());
			scores.push_back(pair<double, int>(predicted_score, labels[item]));
			if (labels[item] == 1){
				num_positives++;
			}
		}

		// Sort the scores ascendingly by the predicted score.
		sort(scores.begin(), scores.end());

		unsigned int TP = num_positives;
		unsigned int FP = sequences.size() - num_positives;
		unsigned int FN = 0;
		unsigned int TN = 0;

		unsigned int min_error = FP + FN;
		unsigned int current_error = 0;
		double best_threshold = -numeric_limits<double>::max();

		for (unsigned int i = 0; i < sequences.size(); ++i) {
			// Take only 1st in a string of equal values
			if (i != 0 && scores[i].first > scores[i-1].first) {
				current_error = FP + FN; // sum of errors, e.g # training errors
				if (current_error < min_error) {
					min_error = current_error;
					best_threshold = (scores[i-1].first + scores[i].first) / 2;
					//cout << "\nThreshold: " << best_threshold;
					//cout << "\n# errors (FP + FN): " << min_error;
					//std::printf ("\nAccuracy: %.5f%% (%d/%d)\n", 100.0 * (TP + TN) / all, TP + TN, all);
				}
			}
			if (scores[i].second == 1) {
				FN++; TP--;
			}else{
				FP--; TN++;
			}
		}

		return best_threshold;

	}

	void predict(vector<string> &test_samples, double threshold = 0){
		test_scores.clear();
		for (unsigned int item = 0; item < test_samples.size(); ++item){
			double predicted_score = brute_classify(test_samples[item].c_str());
			test_scores.push_back(predicted_score);
		}
	}

	double classify(vector<string> &test_samples, vector<double> &test_labels, double threshold = 0){
		test_scores.clear();
		unsigned int verbose = 0;
		unsigned int all = 0;
		unsigned int correct = 0;

		for (unsigned int item = 0; item < test_samples.size(); ++item){
			//int item = test_fold[ic];
			int y = int (test_labels[item]);
			double predicted_score = brute_classify(test_samples[item].c_str());
			test_scores.push_back(predicted_score);

			all++;
			if (predicted_score > 0) {
				if(y > 0) correct++;
				//if(y > 0) res_a++; else res_b++;
			} else {
				if(y < 0) correct++;
				//if(y > 0) res_c++; else res_d++;
			}

		}


		return 1.0 - 1.0*correct / all;

	}

	double run_sax_seql(vector<string>& train_sequences, vector<double>& train_labels, vector<string>& test_sequences, vector<double>& test_labels){

		double threshold = 0.0;

		// ******************************LEARNING PHASE******************************
		clock_t learn_starttime = clock();
		learn(train_sequences, train_labels);

		//std::map<std::string, double>::iterator it = model.begin();
		//while(it != model.end())
		//{
		//	cout << it->first << endl;
		//	it++;
		//}
		//cout << "Learn ok" << endl;

		lastrun_learn_time = double(clock() - learn_starttime) / CLOCKS_PER_SEC;
		//std::map<std::string, double>::iterator it = model.begin();
		// ******************************TUNE THRESHOLD******************************
		// threshold = tune_threshold(train_sequences, train_labels);
		// ******************************CLASSIFICATION PHASE******************************
		clock_t classify_starttime = clock();
		double error = classify(test_sequences,test_labels,threshold);
		//cout << "Test ok" << endl;
		lastrun_classification_time = double(clock() - classify_starttime) / CLOCKS_PER_SEC;

		return error;
	}

	void to_feature_space(vector<string>& sequences, vector<vector<int>>& fs){
		//fs.clear();
		int row = 0;
		for (string sq: sequences){
			while (row >= fs.size()){
				fs.push_back(vector<int>());
			}
			std::map<std::string, double>::iterator it = model.begin();
			while(it != model.end())
			{
				if (contain_pattern(sq,it->first)){
					fs[row].push_back(1);
				} else {
					fs[row].push_back(0);
				}
				it++;
			}
			row++;
		}

	}

	// extract positive features only
	void to_positive_feature_space(vector<string>& sequences, vector<vector<int>>& fs){
		//fs.clear();
		int row = 0;
		for (string sq: sequences){
			std::map<std::string, double>::iterator it = model.begin();
			while (row >= fs.size()){
				fs.push_back(vector<int>());
			}
			while(it != model.end())
			{
				if (it->second > 0) {
					if (contain_pattern(sq,it->first)){
						fs[row].push_back(1);
					} else {
						fs[row].push_back(0);
					}
				}
				it++;
			}
			row++;
		}

	}

	vector<double>* get_test_scores(){
		return &test_scores;
	}

	double get_last_learn_time(){
		return lastrun_learn_time;
	}

	double get_last_classification_time(){
		return lastrun_classification_time;
	}

	vector<string> get_sequence_features(bool only_positive){
		std::map<std::string, double>::iterator it = model.begin();
		vector<string> sf;
		while(it != model.end())
		{
			//cout << it->first << "::" << it->second << endl;
			if (!only_positive || it->second > 0){
				sf.push_back(it->first);
			}
			it++;
		}
		return sf;
	}

	vector<double> get_coefficients(bool only_positive){
		std::map<std::string, double>::iterator it = model.begin();
		vector<double> sf;
		while(it != model.end())
		{
			//cout << it->first << "::" << it->second << endl;
			if (!only_positive || it->second > 0){
				sf.push_back(it->second);
			}
			it++;
		}
		return sf;
	}

	void print_model(int first_k){
		std::map<std::string, double>::iterator it = model.begin();
		while(it != model.end() && first_k > 0)
		{
			cout << it->first << "::" << it->second << endl;
			it++;
			first_k--;
		}
	}
};

#endif /* SEQL_H_ */
