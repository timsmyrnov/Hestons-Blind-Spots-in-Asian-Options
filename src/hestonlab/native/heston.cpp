// C++17 direct arithmetic-Asian Heston Monte Carlo. No learned pricing model.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct Event { double time, forward; int contract; };
struct Step { double dt, sqrt_dt; std::vector<Event> events; };
void require(bool ok, const char* why) { if (!ok) throw std::invalid_argument(why); }
}

extern "C" const char* heston_error() { return last_error.c_str(); }

extern "C" int heston_price(
    int count, const int* offsets, const double* times, const double* forwards,
    const int* totals, const double* known, const double* strikes,
    const int* sides, const double* discounts, const double* p,
    int paths, std::uint64_t seed, double max_step, int full_covariance,
    double* means, double* errors, double* covariance) noexcept {
  try {
    require(count > 0 && count <= 10000, "Invalid contract count");
    require(paths >= 4 && paths % 2 == 0, "Paths must be even and >= 4");
    require(std::isfinite(max_step) && max_step > 0, "Invalid time step");
    for (int j=0; j<5; ++j) require(std::isfinite(p[j]), "Nonfinite parameters");
    require(p[0]>0 && p[1]>0 && p[2]>0 && p[3]>0 && std::abs(p[4])<1,
            "Invalid Heston parameters");
    require(offsets[0] == 0, "Invalid offsets");
    std::vector<Event> events;
    for (int c=0; c<count; ++c) {
      require(offsets[c+1]>=offsets[c] && totals[c]>=1 &&
              totals[c]>=offsets[c+1]-offsets[c], "Invalid fixing count");
      require(std::isfinite(known[c]) && std::isfinite(strikes[c]) &&
              std::isfinite(discounts[c]) && discounts[c]>0 &&
              (sides[c]==1 || sides[c]==-1), "Invalid payoff inputs");
      double previous=0;
      for (int i=offsets[c]; i<offsets[c+1]; ++i) {
        require(std::isfinite(times[i]) && times[i]>previous &&
                std::isfinite(forwards[i]) && forwards[i]>0, "Invalid fixing inputs");
        events.push_back({times[i], forwards[i], c}); previous=times[i];
      }
    }
    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
      if (a.time != b.time) return a.time<b.time;
      return a.contract<b.contract;
    });
    std::vector<Step> steps;
    double previous=0;
    for (std::size_t i=0; i<events.size();) {
      double target=events[i].time;
      double raw_steps=std::ceil((target-previous)/max_step);
      require(raw_steps<=1000000 && steps.size()+raw_steps<=1000000,
              "Time grid is too large");
      int n=static_cast<int>(raw_steps);
      double dt=(target-previous)/n;
      for (int j=0;j<n;++j) steps.push_back({dt,std::sqrt(dt),{}});
      while (i<events.size() && events[i].time==target) steps.back().events.push_back(events[i++]);
      previous=target;
    }
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    const double perpendicular=std::sqrt(1-p[4]*p[4]);
    const int pairs=paths/2;
    std::vector<double> sums(2*count), payoff(count), delta(count), m2(count),
                        cross(full_covariance ? count*count : 0, 0.0);
    std::fill(means, means+count, 0.0);
    std::fill(covariance, covariance+count*count, 0.0);
    for (int pair=0;pair<pairs;++pair) {
      double variance[2]={p[0],p[0]}, factor[2]={1,1};
      for (int c=0;c<count;++c) sums[c]=sums[count+c]=known[c];
      for (const auto& step:steps) {
        double z0=normal(rng), z1=normal(rng);
        double shock=p[4]*z0+perpendicular*z1;
        for (int a=0;a<2;++a) {
          double sign=a ? -1 : 1, positive=std::max(variance[a],0.0);
          double diffusion=std::sqrt(positive)*step.sqrt_dt;
          factor[a]*=std::exp(-0.5*positive*step.dt+diffusion*sign*shock);
          variance[a]+=p[1]*(p[2]-positive)*step.dt+p[3]*diffusion*sign*z0;
          require(std::isfinite(factor[a]) && std::isfinite(variance[a]),
                  "Nonfinite simulated state");
          for (const auto& event:step.events)
            sums[a*count+event.contract]+=event.forward*factor[a];
        }
      }
      for (int c=0;c<count;++c) {
        double first=std::max(sides[c]*(sums[c]/totals[c]-strikes[c]),0.0);
        double second=std::max(sides[c]*(sums[count+c]/totals[c]-strikes[c]),0.0);
        payoff[c]=discounts[c]*(first+second)/2;
        require(std::isfinite(payoff[c]), "Nonfinite simulated payoff");
        delta[c]=payoff[c]-means[c];
        means[c]+=delta[c]/(pair+1);
        m2[c]+=delta[c]*(payoff[c]-means[c]);
      }
      if (full_covariance)
        for (int c=0;c<count;++c) for (int d=0;d<=c;++d)
          cross[c*count+d]+=delta[c]*(payoff[d]-means[d]);
    }
    const double denom=static_cast<double>(pairs)*(pairs-1);
    for (int c=0;c<count;++c) {
      errors[c]=std::sqrt(std::max(m2[c]/denom,0.0));
      covariance[c*count+c]=errors[c]*errors[c];
      if (full_covariance) for (int d=0;d<c;++d)
        covariance[c*count+d]=covariance[d*count+c]=cross[c*count+d]/denom;
    }
    last_error.clear();
    return 0;
  } catch (const std::exception& e) { last_error=e.what(); return 1; }
  catch (...) { last_error="Unknown C++ failure"; return 2; }
}
