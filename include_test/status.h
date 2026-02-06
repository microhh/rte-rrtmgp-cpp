/*
 * This file is imported from MicroHH (https://github.com/earth-system-radiation/earth-system-radiation)
 * and is adapted for the testing of the C++ interface to the
 * RTE+RRTMGP radiation code.
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef STATUS_H
#define STATUS_H

#include <sstream>
#include <string>
#include <iostream>


namespace Status
{
    inline void print_message(const std::ostringstream& ss)
    {
        std::cout << ss.str();
    }

    inline void print_message(const std::string& s)
    {
        std::cout << s << std::endl;
    }

    inline void print_warning(const std::ostringstream& ss)
    {
        std::cout << "WARNING: " << ss.str();
    }

    inline void print_warning(const std::string& s)
    {
        std::cout << "WARNING: " << s << std::endl;
    }

    inline void print_error(const std::ostringstream& ss)
    {
        std::cerr << "ERROR: " << ss.str();
    }

    inline void print_error(const std::string& s)
    {
        std::cerr << "ERROR: " << s << std::endl;
    }
}
#endif
