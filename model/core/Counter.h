
#pragma once

#include "main_header.h"

namespace core {

  // A simple counter that tracks elapsed time and checks if a specified frequency has been reached
  // This is for recurrent tasks that need to be performed at regular time intervals
  struct Counter {
    double etime;
    double freq;

    // Constructor to initialize the counter with a frequency and optional initial elapsed time
    Counter(double freq = 1, double etime = 0) {this->freq = freq; this->etime = etime; }

    // Update the elapsed time by dt and check if the frequency has been reached
    // Returns true if the frequency is reached, false otherwise
    bool update_and_check( double dt ) { etime += dt; return check(); }

    // Update the elapsed time by dt
    void update( double dt ) { etime += dt; }

    // Check if the elapsed time has reached or exceeded the frequency
    // Returns true if the frequency is reached, false otherwise
    bool check() const { return etime >= freq - 1.e-10; }

    // Reset the elapsed time by subtracting the frequency
    void reset() { etime -= freq; }
  };

}

