/*
 * sax_converter.h
 *
 *  Created on: 6 Jul 2016
 *      Author: thachln
 */

#ifndef SAX_CONVERTER_H_
#define SAX_CONVERTER_H_


#include <iostream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
#include <set>


class SAX
{
private:

	double *break_points;
	// bool divisible;


	int MAX; // max length of the time series
	int numerosity_reduction;

public:
	int window_size;
	int step_size;
	int word_length;
	int alphabet_size;
	// numerosity reduction strategy
	static const int NONE_NR = 0;
	static const int BACK2BACK_NR = 1;
	static const int UNIQUE_SET_NR = 2;

	static const int TEST_NORMALIZE = 1;

	void init(int N, int w, int a, int _step_size, int nr_strategy){
		window_size = N;
		word_length = w;
		alphabet_size = a;
		step_size = _step_size;
		numerosity_reduction = nr_strategy;
		MAX = 10000;
		initialize_break_points();

		// if (window_size % word_length == 0){
		// 	divisible = true;
		// } else {
		// 	divisible = false;
		// }
	}

	// Constructor
	SAX(){
		init(64,16,4,1,NONE_NR);
	}

	SAX(int w, int a){
		init(-1,w,a,1,NONE_NR);
	}

	SAX(int N, int w, int a){
		init(N,w,a,1,BACK2BACK_NR);
	}

	SAX(int N, int w, int a, int nr_strategy){
		init(N,w,a,1,nr_strategy);
	}

	SAX(int N, int w, int a, int _step_size, int nr_strategy){
		init(N,w,a,_step_size,nr_strategy);
	}

	// Destructor
	~SAX(){
		free(break_points);
	}

	// initialize the values of break points by alphabet size
	void initialize_break_points(){
		break_points = (double*)malloc (sizeof(double)*(alphabet_size-1));

		//double bps[alphabet_size - 1];
		switch (alphabet_size)
		{
		case 2:{
			double bps[1] = { 0.0 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;
		}
		case 3:{
			double bps[2] = { -0.430727299295, 0.430727299295 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;
		}
		case 4:{
			double bps[3] = { -0.674489750196, 0.0, 0.674489750196 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;
		}
		case 5:{
			double bps[4] = { -0.841621233573, -0.253347103136, 0.253347103136, 0.841621233573 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;
		}
		case 6:{
			double bps[5] = { -0.967421566102, -0.430727299295, 0.0, 0.430727299295, 0.967421566102 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;
		}
		case 7:{
			double bps[6] = { -1.06757052388, -0.565948821933, -0.180012369793, 0.180012369793, 0.565948821933, 1.06757052388 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		case 8:{
			double bps[7] = { -1.15034938038, -0.674489750196, -0.318639363964, 0.0, 0.318639363964, 0.674489750196, 1.15034938038 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		case 9:{
			double bps[8] = { -1.22064034885, -0.764709673786, -0.430727299295, -0.139710298882, 0.139710298882, 0.430727299295, 0.764709673786, 1.22064034885 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		case 10:{
			double bps[9] = { -1.28155156554, -0.841621233573, -0.524400512708, -0.253347103136, 0.0, 0.253347103136, 0.524400512708, 0.841621233573, 1.28155156554 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		case 11:{
			double bps[10] = { -1.33517773612, -0.908457868537, -0.604585346583, -0.348755695517, -0.114185294321, 0.114185294321, 0.348755695517, 0.604585346583, 0.908457868537, 1.33517773612 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		case 12:{
			double bps[11] = { -1.3829941271, -0.967421566102, -0.674489750196, -0.430727299295, -0.210428394248, 0.0, 0.210428394248, 0.430727299295, 0.674489750196, 0.967421566102, 1.3829941271 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		case 13:{
			double bps[12] = { -1.42607687227, -1.02007623279, -0.736315917376, -0.502402223373, -0.293381232121, -0.0965586152896, 0.0965586152896, 0.293381232121, 0.502402223373, 0.736315917376, 1.02007623279, 1.42607687227 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		case 14:{
			double bps[13] = { -1.46523379269, -1.06757052388, -0.791638607743, -0.565948821933, -0.366106356801, -0.180012369793, 0.0, 0.180012369793, 0.366106356801, 0.565948821933, 0.791638607743, 1.06757052388, 1.46523379269 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		case 15:{
			double bps[14] = { -1.50108594604, -1.11077161664, -0.841621233573, -0.62292572321, -0.430727299295, -0.253347103136, -0.0836517339071, 0.0836517339071, 0.253347103136, 0.430727299295, 0.62292572321, 0.841621233573, 1.11077161664, 1.50108594604 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		case 16:{
			double bps[15] = { -1.53412054435, -1.15034938038, -0.887146559019, -0.674489750196, -0.488776411115, -0.318639363964, -0.15731068461, 0.0, 0.15731068461, 0.318639363964, 0.488776411115, 0.674489750196, 0.887146559019, 1.15034938038, 1.53412054435 };
			std::copy(bps,bps + alphabet_size - 1, break_points);
			break;}
		default:
			std::cout << "" << std::endl;
			break;
		}
		/*
17{ -1.56472647136, -1.18683143276, -0.928899491647, -0.721522283982, -0.541395085129, -0.377391943829, -0.22300783094, -0.0737912738083, 0.0737912738083, 0.22300783094, 0.377391943829, 0.541395085129, 0.721522283982, 0.928899491647, 1.18683143276, 1.56472647136 };
18{ -1.59321881802, -1.22064034885, -0.967421566102, -0.764709673786, -0.58945579785, -0.430727299295, -0.282216147063, -0.139710298882, 0.0, 0.139710298882, 0.282216147063, 0.430727299295, 0.58945579785, 0.764709673786, 0.967421566102, 1.22064034885, 1.59321881802 };
{ -1.61985625864, -1.25211952027, -1.00314796766, -0.80459638036, -0.63364000078, -0.479505653331, -0.336038140372, -0.199201324789, -0.0660118123758, 0.0660118123758, 0.199201324789, 0.336038140372, 0.479505653331, 0.63364000078, 0.80459638036, 1.00314796766, 1.25211952027, 1.61985625864 };
{ -1.64485362695, -1.28155156554, -1.03643338949, -0.841621233573, -0.674489750196, -0.524400512708, -0.385320466408, -0.253347103136, -0.125661346855, 0.0, 0.125661346855, 0.253347103136, 0.385320466408, 0.524400512708, 0.674489750196, 0.841621233573, 1.03643338949, 1.28155156554, 1.64485362695 };
{ -1.66839119395, -1.30917171679, -1.06757052388, -0.876142849247, -0.712443032389, -0.565948821933, -0.430727299295, -0.302980448056, -0.180012369793, -0.0597170997853, 0.0597170997853, 0.180012369793, 0.302980448056, 0.430727299295, 0.565948821933, 0.712443032389, 0.876142849247, 1.06757052388, 1.30917171679, 1.66839119395 };
{ -1.69062162958, -1.33517773612, -1.09680356209, -0.908457868537, -0.747858594763, -0.604585346583, -0.472789120992, -0.348755695517, -0.229884117579, -0.114185294321, 0.0, 0.114185294321, 0.229884117579, 0.348755695517, 0.472789120992, 0.604585346583, 0.747858594763, 0.908457868537, 1.09680356209, 1.33517773612, 1.69062162958 };
{ -1.71167530651, -1.35973738394, -1.12433823157, -0.938814316877, -0.781033811523, -0.640666889919, -0.511936213871, -0.391196258189, -0.275921063108, -0.164210777079, -0.0545189148481, 0.0545189148481, 0.164210777079, 0.275921063108, 0.391196258189, 0.511936213871, 0.640666889919, 0.781033811523, 0.938814316877, 1.12433823157, 1.35973738394, 1.71167530651 };
		 */
	}

	std::string segment2SAX(std::vector<double> &timeseries, int cur_pos, char char_start){
		int window_end = cur_pos + window_size - 1;

		// calculate mean and std
		double mean_wd = 0.0;
		double var_wd = 0.0;
		double sum_wd = 0.0;
		double sumsq_wd = 0.0;
		for (int i = cur_pos; i <= window_end; i++){
			sum_wd += timeseries[i];
			sumsq_wd += timeseries[i]*timeseries[i];
		}

		mean_wd = sum_wd / window_size;
		var_wd = sumsq_wd / window_size - mean_wd*mean_wd;
		//std_wd = sqrt(var_wd / window_size - mean_wd*mean_wd);

		// z-normalize


		std::vector<double> subsection(window_size);
		for (int i = cur_pos; i <= window_end; i++){
			double normalized_value;
			if (TEST_NORMALIZE){
				normalized_value = (timeseries[i] - mean_wd);
				if (var_wd > 0){
					normalized_value = normalized_value / sqrt(var_wd);
				}
			} else {
				normalized_value = timeseries[i];
			}

			// for (int j = (i - cur_pos)*word_length;j < (i - cur_pos)*word_length + word_length; j++){
			subsection[i - cur_pos] = normalized_value;
			// }
		}
		double paa_size = window_size * 1.0 / word_length;
		double windowStartIdx, windowEndIdx, startFraction, endFraction, aggValue,PAA;
		int fullWindowStartIdx, fullWindowEndIdx;
		std::string sax_word = "";
		for (int i = 0; i < word_length; i++) {
			int bin = 0;
			windowStartIdx = paa_size * i;
			windowEndIdx = paa_size * (i+1) - 1;
			fullWindowStartIdx = std::ceil(windowStartIdx);
			fullWindowEndIdx = std::floor(windowEndIdx);
			startFraction = fullWindowStartIdx - windowStartIdx;
			endFraction = windowEndIdx - fullWindowEndIdx;
			aggValue = 0;
			if (startFraction > 0) {
			aggValue += subsection[fullWindowStartIdx - 1] * startFraction;
			}
			// 2nd condition is a safeguard against index-out-of-bounds because of
			// numerical imprecision
			if (endFraction > 0 && fullWindowEndIdx < window_size - 1) {
			aggValue += subsection[fullWindowEndIdx + 1] * endFraction;
			}
			for (int j = fullWindowStartIdx; j <= fullWindowEndIdx; j++) {
			aggValue += subsection[j];
			}
			PAA = aggValue / paa_size;

			for (int j = 0; j < alphabet_size - 1;j++){
				if (PAA >= break_points[j]){
					bin++;
				}
			}
			sax_word += char_start + bin;
		}

		return sax_word;

	}

	//main function to convert a vector of double to a vector of SAX sequences
	std::vector<std::string> timeseries2SAX(std::vector<double> &timeseries, char char_start){

		// length of the time series
		int ts_length = timeseries.size();
		std::vector<std::string>  sax_seqc;
		// sliding windows

		for (int cur_pos = 0; cur_pos < ts_length - window_size + 1; cur_pos += step_size){



			std::string sax_word = segment2SAX(timeseries, cur_pos, char_start);

			switch(numerosity_reduction){
			case BACK2BACK_NR:
				if (sax_seqc.empty() || (sax_seqc.back() != sax_word)){
					sax_seqc.push_back(sax_word);
				}
				break;
			case UNIQUE_SET_NR:
			{
				bool first_appear = true;
				for (int i = 0; i < sax_seqc.size();i++){
					if (sax_seqc[i] == sax_word){
						first_appear = false;
						break;
					}
				}
				if (first_appear){
					sax_seqc.push_back(sax_word);
				}
				break;
			}
			default: // also NONE_NR
				sax_seqc.push_back(sax_word);
				break;
			}
		}
		return sax_seqc;
	}

	// same but always use 'abc..' for the alphabet
	std::vector<std::string> timeseries2SAX(std::vector<double> &timeseries){
		char char_start = 'a';
		return timeseries2SAX(timeseries,char_start);
	}



	// set of patterns instead of a single pattern
	// normalize (spread) scores

	std::vector<std::set<int>> map_patterns(std::vector<double> timeseries,std::vector<std::string> patterns){
		char char_start = 'a';
		// length of the time series
		int ts_length = timeseries.size();
		std::vector<std::string>  sax_seqc;


		std::vector<std::set<int>> marked; // store positions in time series corresponded to the pattern
		for (int i = 0; i < patterns.size();i++){
			marked.push_back(std::set<int>());
		}


		for (int cur_pos = 0; cur_pos < ts_length - window_size + 1; cur_pos += step_size){

			std::string sax_word = segment2SAX(timeseries, cur_pos, char_start);
			// NOTE: should be replaced with more efficient string matching algorithm

			for(int p = 0; p < patterns.size();p++){

				std::string pattern = patterns[p];
				//double pt_score = pt_scores[p];
				size_t pos = 0;
				//std::cout << sax_word << ":" << cur_pos << std::endl;
				while(pos < word_length){
					pos = sax_word.find(pattern,pos);

					if (pos == std::string::npos){
						break;
					} else {
						//std::cout << "found:" << pattern << " at " << pos << std::endl;
						double stt = cur_pos + pos*window_size*1.0/word_length; // starting pst of the subsequence in the time series
						double end = stt + pattern.size()*window_size*1.0/word_length - 1; // end pst of the subsequence in the time series
						int ceil_stt = std::ceil(stt);
						int fl_end = std::floor(end);

						for (int j = ceil_stt; j <= fl_end;j++){
							marked[p].insert(j);
							//accu_score[j] += pt_score; // add score for the subsequence of time series
							//if (accu_score[j] > 1000)
							//		std::cout << accu_score[j] << " ";
						}
						if (ceil_stt > 0){
							marked[p].insert(ceil_stt - 1);
							//accu_score[ceil_stt - 1] += (ceil_stt - stt)*pt_score; // add score to the leftmost of the subsequence
						}
						if(fl_end < timeseries.size() - 1){
							marked[p].insert(fl_end + 1);
							//accu_score[fl_end + 1] += (end - fl_end)*pt_score; // add score to the rightmost of the subsequence
						}
						pos++;
					}
				}

			}

		}

		return marked;

	}


	std::vector<double> map_weighted_patterns(std::vector<double> timeseries,std::vector<std::string> patterns, std::vector<double> pt_scores){
		std::vector<std::set<int>> marked = map_patterns(timeseries, patterns);
		std::vector<double> mapped = std::vector<double>(timeseries.size(),0.0);
		//return accu_score;
		for (int i= 0; i < marked.size(); i++){
			for (int pos : marked[i]){
				mapped[pos] += pt_scores[i] / marked[i].size();
			}
		}
		return mapped;
	}

};



#endif /* SAX_CONVERTER_H_ */
