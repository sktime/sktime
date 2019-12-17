#ifndef FREQT_COMMON_H
#define FREQT_COMMON_H

#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <sstream>

namespace constants
{
	const int DEFAULT_MIN_WINDOW_SIZE = 16;
	const int DEFAULT_WORD_LENGTH = 16;
	const int DEFAULT_ALPHABET_SIZE = 4;

	const std::string TIME_SERIES_DELIMITER = ",";
}

template <class Iterator>
static inline unsigned int tokenize (char *str, char *del, Iterator out, unsigned int max)
{
  char *stre = str + strlen (str);
  char *dele = del + strlen (del);
  unsigned int size = 1;

  while (size < max) {
    char *n = std::find_first_of (str, stre, del, dele);
    *n = '\0';
    *out++ = str;
    ++size;
    if (n == stre) break;
    str = n + 1;
  }
  *out++ = str;

  return size;
}


std::vector<double> string_to_double_vector(std::string str,std::string delimiter){
	std::vector<double> numeric_ts;
	size_t pos = 0;
	std::string token;

	while ((pos = str.find(delimiter)) != std::string::npos) {
		token = str.substr(0, pos);
		//std::cout << token << " ";
		numeric_ts.push_back(atof(token.c_str()));
		str.erase(0, pos + delimiter.length());
	}
	if (!str.empty()){
		numeric_ts.push_back(atof(str.c_str()));
	}
	return numeric_ts;
}

std::vector<int> string_to_int_vector(std::string str,std::string delimiter){
	std::vector<int> numeric_ts;
	size_t pos = 0;
	std::string token;

	while ((pos = str.find(delimiter)) != std::string::npos) {
		token = str.substr(0, pos);
		//std::cout << token << " ";
		numeric_ts.push_back(atoi(token.c_str()));
		str.erase(0, pos + delimiter.length());
	}
	if (!str.empty()){
		numeric_ts.push_back(atoi(str.c_str()));
	}
	return numeric_ts;
}

template <class T>
void print_vector_of_vector(std::vector<std::vector<T>> input){
	for(std::vector<T> vt: input){
		for(T v: vt){
			std::cout << v << " ";
		}
		std::cout << std::endl;
	}
}

template <typename T>
std::string join(const T& v, const std::string& delim) {
	std::ostringstream s;
    for (const auto& i : v) {
        if (&i != &v[0]) {
            s << delim;
        }
        s << i;
    }
    return s.str();
}


#endif
