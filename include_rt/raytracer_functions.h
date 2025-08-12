#ifndef RAYTRACER_FUNCTIONS_H
#define RAYTRACER_FUNCTIONS_H

#include <iostream>
#include <curand_kernel.h>

#include "types.h"
#include "raytracer_definitions.h"
#include "raytracer_functions.h"


namespace Raytracer_functions
{
    using namespace Raytracer_definitions;

    template<typename T> static inline __host__ __device__
    Vector<T> operator*(const Vector<T> v, const Float s) { return Vector<T>{s*v.x, s*v.y, s*v.z}; }
    template<typename T> static inline __host__ __device__
    Vector<T> operator*(const Float s, const Vector<T> v) { return Vector<T>{s*v.x, s*v.y, s*v.z}; }
    template<typename T> static inline __host__ __device__
    Vector<T> operator-(const Vector<T> v1, const Vector<T> v2) { return Vector<T>{v1.x-v2.x, v1.y-v2.y, v1.z-v2.z}; }
    template<typename T> static inline __host__ __device__
    Vector<T> operator-(const Vector<T> v, const Float s) { return Vector<T>{v.x-s, v.y-s, v.z-s}; }
    template<typename T> static inline __host__ __device__
    Vector<T> operator+(const Vector<T> v, const Float s) { return Vector<T>{s+v.x, s+v.y, s+v.z}; }
    template<typename T> static inline __host__ __device__
    Vector<T> operator+(const Vector<T> v1, const Vector<T> v2) { return Vector<T>{v1.x+v2.x, v1.y+v2.y, v1.z+v2.z}; }
    template<typename T> static inline __host__ __device__
    Vector<T> operator/(const Vector<T> v, const Float s) { return Vector<T>{v.x/s, v.y/s, v.z/s}; }
    template<typename T> static inline __host__ __device__
    Vector<T> operator/(const Float s, const Vector<T> v) { return Vector<T>{v.x/s, v.y/s, v.z/s}; }
    template<typename T> static inline __host__ __device__
    Vector<T> operator*(const Vector<T> v1, const Vector<T> v2) { return Vector<T>{v1.x*v2.x, v1.y*v2.y, v1.z*v2.z}; }
    template<typename T> static inline __host__ __device__
    Vector<T> operator/(const Vector<T> v1, const Vector<T> v2) { return Vector<T>{v1.x/v2.x, v1.y/v2.y, v1.z/v2.z}; }

    static inline __host__ __device__
    Vector<Float> operator/(const Vector<Float> v1, const Vector<int> v2) { return Vector<Float>{v1.x/v2.x, v1.y/v2.y, v1.z/v2.z}; }
    static inline __host__ __device__
    Vector<Float> operator*(const Vector<Float> v1, const Vector<int> v2) { return Vector<Float>{v1.x*v2.x, v1.y*v2.y, v1.z*v2.z}; }

    template<typename T> __host__ __device__
    Vector<T> cross(const Vector<T> v1, const Vector<T> v2)
    {
        return Vector<T>{
                v1.y*v2.z - v1.z*v2.y,
                v1.z*v2.x - v1.x*v2.z,
                v1.x*v2.y - v1.y*v2.x};
    }

    template<typename T> __host__ __device__
    Float dot(const Vector<T>& v1, const Vector<T>& v2)
    {
        return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
    }

    template<typename T> __host__ __device__
    Float norm(const Vector<T> v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }


    template<typename T> __host__ __device__
    Vector<T> normalize(const Vector<T> v)
    {
        const Float length = norm(v);
        return Vector<T>{ v.x/length, v.y/length, v.z/length};
    }

    __device__
    inline Float pow2(const Float d) { return d*d; }

    __device__
    inline Float rayleigh(const Float random_number)
    {
        const Float q = Float(4.)*random_number - Float(2.);
        const Float d = Float(1.) + pow2(q);
        const Float u = pow(-q + sqrt(d), Float(1./3.));
        return u - Float(1.)/u;
    }

    __device__
    inline Float henyey(const Float g, const Float random_number)
    {
        const Float a = pow2(Float(1.) - pow2(g));
        const Float b = Float(2.)*g*pow2(Float(2.)*random_number*g + Float(1.) - g);
        const Float c = -g/Float(2.) - Float(1.)/(Float(2.)*g);
        return Float(-1.)*(a/b) - c;
    }

    __device__
    inline int find_index(const float* mie_cdf, const int size, const float random_number)
    {
        int left = 0;
        int right = size - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (random_number >= mie_cdf[mid]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return left - 1;
    }

    __device__
    inline Float mie_sample_angle(const Float* mie_cdf, const Float* mie_lut, const Float random_number, const Float r_eff, const int n_mie)
    {
        // interpolation over effective radius. Currently, r_eff should range between 2.5 and 21.5 (similar to RRTMGP)
        const int r_idx = min(max(int(r_eff-2.5), 0), 18);
        const Float r_rest = fmod(r_eff-Float(2.5),Float(1.));

        const int i = min(max(0, find_index(mie_cdf, n_mie, random_number)), n_mie - 2);

        const int midx_lwr = r_idx*n_mie;
        const int midx_upr = (r_idx+1)*n_mie;
        const Float dr = abs(mie_cdf[i+1] - mie_cdf[i]);

        const Float ang_lwr = (abs(random_number - mie_cdf[i+1])*mie_lut[(i)+midx_lwr] + abs(mie_cdf[i]-random_number)*mie_lut[i+midx_lwr+1]) / dr;
        const Float ang_upr = (abs(random_number - mie_cdf[i+1])*mie_lut[(i)+midx_upr] + abs(mie_cdf[i]-random_number)*mie_lut[i+midx_upr+1]) / dr;
        const Float ang = ang_lwr * (1-r_rest) + ang_upr * r_rest;
        return ang;
    }

    __device__
    inline Float mie_interpolate_phase_table(const Float* mie_phase, const Float* mie_lut, const Float scat_ang, const Float r_eff, const int n_mie)
    {
        // interpolation over effective radius. Currently, r_eff should range between 2.5 and 21.5 (similar to RRTMGP)
        const int r_idx = min(max(int(r_eff-2.5), 0), 18);
        const Float r_rest = fmod(r_eff-Float(2.5),Float(1.));

        // interpolation between 1800 equally spaced scattering angles between 0 and PI (both inclusive).
        constexpr Float d_pi = Float(1.74629942e-03);
        const int i = min(max(0, int(scat_ang/d_pi)), 1798);

        const int midx_lwr = r_idx*n_mie;
        const int midx_upr = (r_idx+1)*n_mie;
        const Float dr = abs(mie_phase[i+1] - mie_phase[i]);

        const Float prob_lwr = (abs(scat_ang - mie_phase[i+1])*mie_lut[(i)+midx_lwr] + abs(mie_phase[i]-scat_ang)*mie_lut[i+1+midx_lwr]) / dr;
        const Float prob_upr = (abs(scat_ang - mie_phase[i+1])*mie_lut[(i)+midx_upr] + abs(mie_phase[i]-scat_ang)*mie_lut[i+1+midx_upr]) / dr;
        const Float prob = prob_lwr * (1-r_rest) + prob_upr * r_rest;

        return prob;
    }

    __device__
    inline Float sample_tau(const Float random_number)
    {
        // Prevent log(0) possibility.
        return Float(-1.)*log(-random_number + Float(1.) + Float_epsilon);
    }

    __device__
    inline int float_to_int(const Float s_size, const Float ds, const int ntot_max)
    {
        const int ntot = static_cast<int>(s_size / ds);
        return ntot < ntot_max ? ntot : ntot_max-1;
    }

    __device__
    inline void write_photon_out(Float* field_out, const Float w)
    {
        #ifdef __CUDACC__
        atomicAdd(field_out, w);
        #endif
    }


    template<typename T>
    struct Random_number_generator
    {
        __device__ Random_number_generator(unsigned int tid)
        {
            curand_init(tid, tid, 0, &state);
        }

        __device__ T operator()();

        curandState state;
    };


    template<>
    __device__ inline double Random_number_generator<double>::operator()()
    {
        return 1. - curand_uniform_double(&state);
    }


    template<>
    __device__ inline float Random_number_generator<float>::operator()()
    {
        return 1.f - curand_uniform(&state);
    }
}
#endif
