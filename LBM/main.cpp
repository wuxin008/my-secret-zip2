#include "LBM.h"

/*void setup(LBM& lbm) { // two rolling colliding droplets in force field
    uint Nx = 128;
    uint Ny = 128;
    uint Nz = 128;
    lbm.init(Nx, Ny, Nz, 0.014f, 0.0001f);

    auto sphere = [](const uint x, const uint y, const uint z, const glm::vec3& p, const float r) {
        const glm::vec3 t = glm::vec3(x, y, z) - p;
        return sq(t.x) + sq(t.y) + sq(t.z) <= sq(r);
        };

    parallel_for(lbm.get_Nxyz(), [&](uint32_t index) {
        uint x = index % Nx, y = (index - x) / Nx % Ny, z = index / Nx / Ny;

        if (sphere(x, y, z, glm::vec3(Nx / 4u, Ny / 4u, Nz / 4u), 2 * Nx / 16)) {
            lbm.flags[index] = TYPE_F;
            lbm.vels[index] = 0.045f;
            lbm.vels[lbm.get_Nxyz() + index] = 0.045f;
        }
        if (sphere(x, y, z, glm::vec3(3 * Nx / 4u, 3 * Ny / 4u, 3 * Nz / 4u), 1 * Nx / 8)) {
            lbm.flags[index] = TYPE_F;
            lbm.vels[index] = -0.045f;
            lbm.vels[lbm.get_Nxyz() + index] = -0.045f;
        }

        lbm.cfs[index] = -0.001f * ((x + 0.5f) / Nx - 0.5f);
        lbm.cfs[lbm.get_Nxyz() + index] = -0.001f * ((y + 0.5f) / Ny - 0.5f);
        lbm.cfs[2 * lbm.get_Nxyz() + index] = -0.001f * ((z + 0.5f) / Nz - 0.5f);

        if (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == 0u || z == Nz - 1u) lbm.flags[index] = TYPE_S;
        });
}/**/

/*void setup(LBM& lbm) { // MOVING BOUNDARY CONDITION TEST
    uint Nx = 125;
    uint Ny = 125;
    uint Nz = 125;
    lbm.init(Nx, Ny, Nz, 0.005f, 1e-6f);

    parallel_for(lbm.get_Nxyz(), [&](uint32_t index) {
        uint x = index % Nx, y = (index - x) / Nx % Ny, z = index / Nx / Ny;

        if (z < Nz * 6u / 8u && y < Ny / 8u) lbm.flags[index] = TYPE_F;
        if (y == 0u) lbm.vels[2u * lbm.get_Nxyz() + index] = -0.1f;
        if (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == 0u || z == Nz - 1u) lbm.flags[index] = TYPE_S;
        });
}/**/

/*void setup(LBM& lbm) { // DAM_BREAK
    uint Nx = 128;
    uint Ny = 128;
    uint Nz = 128;
    lbm.init(Nx, Ny, Nz, 0.2f, 1e-6f);

    parallel_for(lbm.get_Nxyz(), [&](uint32_t index) {
        uint x = index % Nx, y = (index - x) / Nx % Ny, z = index / Nx / Ny;

        if (z < Nz * 6u / 8u && y < Ny / 8u) lbm.flags[index] = TYPE_F;
        //if (x < 5 * Nx / 8u && x > 3 * Nx / 8u && y < 5 * Ny / 8u && y > 3 * Ny / 8u && z < 5 * Nz / 8u && z > 3 * Nz / 8u) lbm.flags[index] = TYPE_F;
        //if (y == 0u) lbm.vels[2u * lbm.get_Nxyz() + index] = -0.1f;
        //if (z == 0u) lbm.vels[2u * lbm.get_Nxyz() + index] = 0.5f;
        if (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == 0u || z == Nz - 1u) lbm.flags[index] = TYPE_S;
    });
}/**/

void setup(LBM& lbm) { // SWIMMING POOL
    uint Nx = 160;
    uint Ny = 96;
    uint Nz = 160;
    lbm.init(Nx, Ny, Nz, 0.2f, 1e-6f);

    parallel_for(lbm.get_Nxyz(), [&](uint32_t index) {
        uint x = index % Nx, y = (index - x) / Nx % Ny, z = index / Nx / Ny;

        if (z < Nz * 5u / 8u && x < Ny / 8u) lbm.flags[index] = TYPE_F;
        //if (x < 5 * Nx / 8u && x > 3 * Nx / 8u && y < 5 * Ny / 8u && y > 3 * Ny / 8u && z < 5 * Nz / 8u && z > 3 * Nz / 8u) lbm.flags[index] = TYPE_F;
        //if (y == 0u) lbm.vels[2u * lbm.get_Nxyz() + index] = -0.1f;
        //if (z == 0u) lbm.vels[2u * lbm.get_Nxyz() + index] = 0.5f;
        if (x == 0u || x == Nx - 1u || y == 0u || y == Ny - 1u || z == 0u || z == Nz - 1u) lbm.flags[index] = TYPE_S;
    });
}/**/

/*void setup(LBM& lbm) { // RIVER
    uint Nx = 128;
    uint Ny = 384;
    uint Nz = 96;
    lbm.init(Nx, Ny, Nz);

    auto cube = [](const uint x, const uint y, const uint z, const glm::vec3& p, const float l) {
        const glm::vec3 t = glm::vec3(x, y, z) - p;
        return t.x >= -0.5f * l && t.x <= 0.5f * l && t.y >= -0.5f * l && t.y <= 0.5f * l && t.z >= -0.5f * l && t.z <= 0.5f * l;
        };
    auto cuboid = [](const uint x, const uint y, const uint z, const glm::vec3& p, const glm::vec3& l) {
        const glm::vec3 t = glm::vec3(x, y, z) - p;
        return t.x >= -0.5f * l.x && t.x <= 0.5f * l.x && t.y >= -0.5f * l.y && t.y <= 0.5f * l.y && t.z >= -0.5f * l.z && t.z <= 0.5f * l.z;
        };
    auto cylinder = [](const uint x, const uint y, const uint z, const glm::vec3& p, const glm::vec3& n, const float r) {
        const glm::vec3 t = glm::vec3(x, y, z) - p;
        const float sqnt = sq(glm::dot(glm::normalize(n), t));
        const float dist = sq(t.x) + sq(t.y) + sq(t.z) - sqnt;
        return dist <= sq(r) && sqnt <= sq(0.5f * glm::length(n));
        };

    parallel_for(lbm.get_Nxyz(), [&](uint32_t index) {
        uint x = index % Nx, y = (index - x) / Nx % Ny, z = index / Nx / Ny;

        const int R = 20, H = 32;
        if (lbm.flags[index] == TYPE_S) return;
        if (z == 0) lbm.flags[index] = TYPE_S;
        else if (z < H) {
            lbm.vels[lbm.get_Nxyz() + index] = -0.1;
            lbm.flags[index] = TYPE_F;
        }
        if (x == 0 || x == Nx - 1u) lbm.flags[index] = TYPE_S;

        if (cylinder(x, y, z, glm::vec3(Nx * 2u / 3u, Ny * 2u / 3u, Nz / 2u) + 0.5f, glm::vec3(0u, 0u, Nz), (float)R))lbm.flags[index] = TYPE_S;
        if (cuboid(x, y, z, glm::vec3(Nx / 3u, Ny / 3u, Nz / 2u) + 0.5f, glm::vec3(2u * R, 2u * R, Nz))) lbm.flags[index] = TYPE_S;
        });
}/**/

int main() {
    LBM lbm;

    setup(lbm);

    lbm.run();

    //try {
    //    lbm.run();
    //}
    //catch (const std::exception& e) {
    //    std::cerr << e.what() << std::endl;
    //    return EXIT_FAILURE;
    //}

    return EXIT_SUCCESS;
}