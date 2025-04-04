#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "Source_functions.h"
#include "tilted_column.h"

void tilted_path(std::vector<Float>& xh, std::vector<Float>& yh,
                 std::vector<Float>& zh, std::vector<Float>& z,
                 Float sza, Float azi,
                 Float x_start, Float y_start,
                 std::vector<ijk>& tilted_path,
                 std::vector<Float>& zh_tilted)
{
    const Float dx = xh[1]-xh[0]; 
    const Float dy = yh[1]-yh[0]; 
    const Float z_top = *std::max_element(zh.begin(), zh.end());
    const int n_x = xh.size()-1;
    const int n_y = yh.size()-1;
    int i = 0;
    int j = 0;
    int k = 0;

    Float xp = xh[0] + x_start*dx;
    Float yp = yh[0] + y_start*dy;
    Float zp = 0.;
    Float dl = 0.;

    tilted_path.clear();
    zh_tilted.clear();
    std::vector<Float> dz_tilted;

    Float dir_x = std::sin(sza) * std::sin(azi); // azi 0 is from the north
    Float dir_y = std::sin(sza) * std::cos(azi);
    Float dir_z = std::cos(sza);  

    int z_idx = -1;
    
    const Float epsilon = 1e-8; // Small value to handle floating-point precision
    const Float min_step = 1e-2; // Minimum step size in meters
    
    tilted_path.push_back({i, j, k}); // Add starting point
    dz_tilted.push_back(0.0);
    z_idx = 0;
    
    while (zp < z_top)
    {           
        // Check bounds before accessing arrays
        if (k + 1 >= zh.size()) {
            std::cerr << "Error: k+1 (" << k + 1 << ") out of bounds for zh (size=" << zh.size() << ")" << std::endl;
            break;
        }
        if (j + 1 >= yh.size()) {
            std::cerr << "Error: j+1 (" << j + 1 << ") out of bounds for yh (size=" << yh.size() << ")" << std::endl;
            break;
        }
        if (i + 1 >= xh.size()) {
            std::cerr << "Error: i+1 (" << i + 1 << ") out of bounds for xh (size=" << xh.size() << ")" << std::endl;
            break;
        }
        
        // Calculate distances to next cell boundaries
        Float lz = (std::abs(dir_z) < epsilon) ? 
                    100000.: (zh[k+1] - zp) / dir_z;
        
        // Handle cases where we're extremely close to a boundary
        if (std::abs(zp - zh[k+1]) < epsilon && dir_z > 0) {
            k += 1;
            zp = zh[k];
            if (k + 1 >= zh.size()) {
                break; // We've reached the top
            }
            continue;
        }
        
        Float ly;
        if (std::abs(dir_y) < epsilon) {
            ly = 100000.;
        } else if (dir_y < 0) {
            if (std::abs(yp - yh[j]) < epsilon) {
                // Already at the lower boundary, move to the previous cell
                j = (j == 0) ? n_y - 1 : j - 1;
                yp = yh[j+1] - epsilon; // Position just inside the cell
                continue;
            }
            ly = (yp - yh[j]) / (-dir_y);
        } else { // dir_y > 0
            if (std::abs(yp - yh[j+1]) < epsilon) {
                // Already at the upper boundary, move to the next cell
                j = (j + 1) % n_y;
                yp = yh[j] + epsilon;
                continue;
            }
            ly = (yh[j+1] - yp) / dir_y;
        }
        
        Float lx;
        if (std::abs(dir_x) < epsilon) {
            lx = 100000.;
        } else if (dir_x < 0) {
            if (std::abs(xp - xh[i]) < epsilon) {
                // Already at the lower boundary, move to the previous cell
                i = (i == 0) ? n_x - 1 : i - 1;
                xp = xh[i+1] - epsilon;
                continue;
            }
            lx = (xp - xh[i]) / (-dir_x);
        } else { // dir_x > 0
            if (std::abs(xp - xh[i+1]) < epsilon) {
                // Already at the upper boundary, move to the next cell
                i = (i + 1) % n_x;
                xp = xh[i] + epsilon;
                continue;
            }
            lx = (xh[i+1] - xp) / dir_x;
        }

        Float l = std::min({lx, ly, lz});
        Float dx0 = l * dir_x;
        Float dy0 = l * dir_y;
        Float dz0 = l * dir_z;
        // Move along axes:
        xp += dx0;
        yp += dy0;
        zp += dz0;
        
        // Record the path segment
        dz_tilted[z_idx] += dz0;
        
        // Check z boundary crossing
        if (std::abs(l - lz) < epsilon || zp >= zh[k+1]) {
            k += 1;
            // Create a new path segment after crossing boundary
            tilted_path.push_back({i, j, k});
            dz_tilted.push_back(0.0);
            z_idx += 1;
        }
        
        // Check y boundary crossing
        if (std::abs(l - ly) < epsilon) {

            j = int(j + sign(dy0));
            j = (j == -1) ? n_y - 1 : j%n_y;
            yp = dy0 < 0 ? yh[j+1] : yh[j];
            
        }
        
        // Check x boundary crossing
        if (std::abs(l - lx) < epsilon) {
            i = int(i + sign(dx0));
            i = (i == -1) ? n_x - 1 : i%n_x;
            xp = dx0 < 0 ? xh[i+1] : xh[i];
        }
    }    
    // Construct final zh_tilted
    zh_tilted.clear();
    zh_tilted.push_back(0.);
    for (int iz = 0; iz < dz_tilted.size(); ++iz) {
        zh_tilted.push_back(zh_tilted[iz] + dz_tilted[iz]);
    }
}
