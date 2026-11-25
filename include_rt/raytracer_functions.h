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
    inline Float fresnel_unpolarized(const Float cos_i, const Float n_rel)
    {
        const Float sin2_t = (Float(1.) - cos_i*cos_i) / (n_rel*n_rel);
        const Float cos_t  = sqrt(max(Float(0.), Float(1.) - sin2_t));
        const Float rs = (cos_i - n_rel*cos_t) / (cos_i + n_rel*cos_t);
        const Float rp = (n_rel*cos_i - cos_t) / (n_rel*cos_i + cos_t);
        return Float(0.5) * (rs*rs + rp*rp);
    }

    enum Surface_scattering : int
    {
        Surface_lambertian          = 0,  // Purely lambertian, albedo = surface_albedo
        Surface_specular            = 1,  // Purely specular, albedo = Fresnel
        Surface_specular_lambertian = 2,  // Specular reflection (Fresnel) with water-leaving fraction (lambertian, albedo = surface_albedo)
    };

    struct Surface_reflectance
    {
        Float R_spec; // specular (Fresnel) reflectance
        Float R_diff; // lambertian reflectance
    };

    // Specular and lambertian surface reflectance for a photon hitting the surface with cos(incidence angle) cos_i
    __device__
    inline Surface_reflectance surface_reflectance(const Surface_props& props, const Float cos_i)
    {
        if (props.scattering_type == Surface_lambertian)
        {
            return {Float(0.), props.albedo};
        }
        else if (props.scattering_type == Surface_specular)
        {
            return {fresnel_unpolarized(cos_i, n_water), Float(0.)};
        }
        else // Surface_specular_lambertian
        {
            const Float R_spec = fresnel_unpolarized(cos_i, n_water);
            return {R_spec, (Float(1.) - R_spec) * props.albedo};
        }
    }

    template<typename T> __device__
    inline Vector<T> specular(const Vector<T> dir_in, const Vector<T> dir_n)
    {
        return dir_in - Float(2.)*dir_n*dot(dir_n, dir_in);
    }

    // minumum sigma2 value use Cox-Munk microfacet distribution, otherwise assume perfectly flat surface
    constexpr Float sigma2_min = Float(1e-5);

    __device__
    inline bool is_rough(const Surface_props& props)
    {
        return (props.scattering_type != Surface_lambertian) && (props.sigma2 > sigma2_min);
    }

    // Beckmann face normal ditribution / isotropic Cox-Munk PDF normalized such that int D cos_n dOmega equals 1
    __device__
    inline Float cox_munk_D(const Float cos_n, const Float sigma2)
    {
        const Float sq_cos_n = cos_n*cos_n;
        const Float sq_tan_n = (Float(1.) - sq_cos_n) / sq_cos_n;

        return exp(-sq_tan_n/sigma2) / (Float(M_PI)*sigma2*sq_cos_n*sq_cos_n);
    }

    // Smith masking function, approximated proposed by Walter et al. (2007) "Microfacet Models for Refraction through Rough Surfaces"
    __device__
    inline Float cox_munk_G(const Float mu, const Float sigma2)
    {
        if (mu <= Float(0.))
            return Float(0.);

        const Float a = mu / max(sqrt(sigma2*(Float(1.) - mu*mu)), Float(1e-12));

        if (a >= Float(1.6))
            return Float(1.);

        return (Float(3.535) * a + Float(2.181) * a * a) / (Float(1.) + Float(2.276) * a + Float(2.577) * a * a);
    }

    // Smith Lambda for Beckmann / isotropic Cox-Munk, from the Walter et al. (2007) eq. 27 fit
    // via Lambda = (1 - G1)/G1 (Heitz 2014, below eq. 69)
    __device__
    inline Float cox_munk_Lambda(const Float mu, const Float sigma2)
    {
        const Float a = mu / max(sqrt(sigma2*(Float(1.) - mu*mu)), Float(1e-12));

        if (a >= Float(1.6))
            return Float(0.);

        return (Float(1.) - Float(1.259) * a + Float(0.396) * a * a) / (Float(3.535) * a + Float(2.181) * a * a);
    }

    // height-correlated Smith masking-shadowing function, Heitz (2014) eq. 99
    __device__
    inline Float cox_munk_G2(const Float mu_in, const Float mu_out, const Float sigma2)
    {
        if (mu_in <= Float(0.) || mu_out <= Float(0.))
            return Float(0.);

        return Float(1.) / (Float(1.) + cox_munk_Lambda(mu_in, sigma2) + cox_munk_Lambda(mu_out, sigma2));
    }

    // cox_munk/beckmann brdf times mu_out with height-correlated Smith shadowing. Walter et al. (2007) eq. 20, Heitz (2014) eq. 99
    __device__
    inline Float cox_munk_brdf_cos(const Vector<Float>& dir_in, const Vector<Float>& dir_out, const Float sigma2)
    {
        const Float mu_in  = -dir_in.z;
        const Float mu_out =  dir_out.z;

        if (mu_in <= Float(0.) || mu_out <= Float(0.))
            return Float(0.);

        const Vector<Float> half = normalize(dir_out - dir_in);
        const Float cos_half = max(dot(dir_out, half), Float(0.));

        return fresnel_unpolarized(cos_half, n_water) * cox_munk_D(half.z, sigma2)
             * cox_munk_G2(mu_in, mu_out, sigma2) / (Float(4.)*mu_in);
    }

    struct Surface_interaction
    {
        Vector<Float> dir_spec; // specular direction: mirror about the (sampled) facet normal
        Float R_spec;           // flat: F(mu_in)               rough: F G2 cos_in/(n_z mu_in)
        Float R_diff;           // flat: (1-F(mu_in)) albedo    rough: (1-F) G1_in cos_in/(n_z mu_in) albedo
    };

    // Partition of a photon hitting the surface into specular and lambertian (water-leaving) parts.
    // Rough surfaces: facet normal n ~ D(n) n_z (Walter et al. 2007, eqs. 28-29), and both parts use the
    // local Fresnel F(cos_in) of that facet, so their expectations are consistent with each other and with
    // cox_munk_brdf_cos. Expectation of R_spec is the directional albedo of the rough specular BRDF.
    template<typename RNG> __device__
    inline Surface_interaction surface_interaction(const Surface_props& sfc, const Vector<Float>& dir_in, RNG& rng)
    {
        const Vector<Float> z_hat = {Float(0.), Float(0.), Float(1.)};
        const Float mu_in = -dir_in.z;

        if (!is_rough(sfc))
        {
            const Surface_reflectance refl = surface_reflectance(sfc, max(mu_in, Float(0.)));
            return {specular(dir_in, z_hat), refl.R_spec, refl.R_diff};
        }

        const Surface_interaction none = {dir_in, Float(0.), Float(0.)};
        if (mu_in <= Float(0.))
            return none;

        // isotropic gaussian slopes, slope^2 ~ Exp(mean sigma2)
        const Float slope = sqrt(-sfc.sigma2 * log(max(rng(), Float_epsilon)));
        const Float phi = Float(2.*M_PI) * rng();
        const Vector<Float> n = normalize(Vector<Float>{-slope*cos(phi), -slope*sin(phi), Float(1.)});

        // back-facing facet: zero visible area (exact zero)
        const Float cos_in = -dot(dir_in, n);
        if (cos_in <= Float(0.))
            return none;

        const Float F    = fresnel_unpolarized(cos_in, n_water);
        const Float geom = cos_in / (n.z * mu_in); // beam-intercepted facet area relative to its footprint
        const Vector<Float> dir_out = specular(dir_in, n);

        // specular: G2 = 0 if reflected into the surface (second bounce not modelled)
        const Float R_spec = F * cox_munk_G2(mu_in, dir_out.z, sfc.sigma2) * geom;

        // lambertian: transmitted through this facet, only shadowed on the way in
        const Float R_diff = (sfc.scattering_type == Surface_specular_lambertian)
                           ? (Float(1.) - F) * cox_munk_G(mu_in, sfc.sigma2) * geom * sfc.albedo
                           : Float(0.);

        return {dir_out, R_spec, R_diff};
    }

    __device__
    inline int find_index(const Float* mie_cdf, const int size, const Float random_number)
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
    inline Float mie_sample_angle(const Float* mie_cdf, const Float* mie_lut, const Float random_number, const Float* r_eff, const int ijk, const int n_mie)
    {
        // interpolation over effective radius. Currently, r_eff should range between 2.5 and 21.5 (similar to RRTMGP)
        const Float r_pos = r_eff[ijk] - Float(2.5);
        const int r_idx = min(max(int(floor(r_pos)), 0), 18);
        const Float r_rest = min(max(r_pos - Float(r_idx), Float(0.)), Float(1.));

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
    inline Float mie_interpolate_phase_table(const Float* mie_phase, const Float* mie_lut, const Float scat_ang, const Float* r_eff, const int ijk, const int n_mie)
    {
        // interpolation over effective radius. Currently, r_eff should range between 2.5 and 21.5 (similar to RRTMGP)
        const Float r_pos = r_eff[ijk] - Float(2.5);
        const int r_idx = min(max(int(floor(r_pos)), 0), 18);
        const Float r_rest = min(max(r_pos - Float(r_idx), Float(0.)), Float(1.));

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
    inline int height_to_int(const Float z, const Float* z_lev, const int* z_lut, const Float lut_dz, const int lut_size, const int ntot_max)
    {
        int k = z_lut[min(max(static_cast<int>(z / lut_dz), 0), lut_size-1)];
        while (k > 0 && z < z_lev[k])
            --k;
        while (k < ntot_max-1 && z >= z_lev[k+1])
            ++k;
        return k;
    }

    template<Bool dz_constant> __device__
    inline int grid_z_index(const Float z, const Float dz, const Float* z_lev,
                            const int* z_lut, const Float lut_dz, const int lut_size, const int ntot_max)
    {
        if constexpr (dz_constant)
            return float_to_int(z, dz, ntot_max);
        else
            return height_to_int(z, z_lev, z_lut, lut_dz, lut_size, ntot_max);
    }

    template<Bool dz_constant> __device__
    inline Float grid_z_bound(const int k, const Float dz, const Float* z_lev)
    {
        if constexpr (dz_constant)
            return k * dz;
        else
            return z_lev[k];
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
        __device__ Random_number_generator(Int tid)
        {
            #ifdef SAFERNG
            curand_init(0ULL, tid, Int(0), &state);
            #else
            curand_init(tid, 0ULL, Int(0), &state);
            #endif
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
