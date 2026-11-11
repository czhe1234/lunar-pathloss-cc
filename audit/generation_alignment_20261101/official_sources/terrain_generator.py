"""
Lunar Terrain Generator (LTGen)

Description:
    A physics-informed synthetic terrain generator for lunar surfaces. This module 
    implements statistical cratering models and topographic diffusion to synthesize 
    realistic Digital Elevation Models (DEMs) of the lunar maria and highlands.

    Key features include:
    - Crater Production: Based on Neukum et al. (2001) production functions.
    - Crater Morphology: Based on meter-scale profiles from Cai & Fa (2020).
    - Erosion: Modeled via topographic diffusion (Fassett & Thomson, 2014).
    - Micro-roughness: Sub-resolution rock abundance and regolith gardening.

References:
    [1] Cai, Y., & Fa, W. (2020). "Meter-scale topographic roughness of the Moon: 
        The effect of small impact craters." Journal of Geophysical Research: Planets, 
        125, e2020JE006429. https://doi.org/10.1029/2020JE006429
    
    [2] Neukum, G., Ivanov, B. A., & Hartmann, W. K. (2001). "Cratering records 
        in the inner solar system in relation to the lunar reference system." 
        Space Science Reviews, 96, 55-86.

    [3] Fassett, C. I., & Thomson, B. J. (2014). "Crater degradation on the lunar 
        maria: Topographic diffusion and the rate of erosion on the Moon." 
        Journal of Geophysical Research: Planets, 119, 2255-2271.

Author: Anders Michael Pearson
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit, prange
import time
from noise import pnoise2
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from tqdm import tqdm




# had to put this method out of the class because 
# numba doesn't work within classes
@njit(parallel=True)
def batch_shift_and_add_numba(heightmap, craters, centers, shifts):
    """
    Add crater profiles to a heightmap in parallel using Numba.
    Heightmap: np.ndarray, heightmap to add craters to
    Craters: list of np.ndarray, craters to add to heightmap
    Centers: list of tuples, location on heightmap to shift crater centers to
    Shifts: list of tuples, location shift for center of crater 
            (necessary to add all parts of craters with profiles
            larger than heightmap)
    """
    H_a, W_a = heightmap.shape
    num_threads = numba.get_num_threads()  
    temp_arrays = np.zeros((num_threads, H_a, W_a))  
    d_scale = np.random.exponential(1, len(craters))
    for i in prange(len(craters)):
        b = craters[i]
        cx, cy = centers[i]
        H_b, W_b = b.shape
        shift_r, shift_c = cy - H_b // 2, cx - W_b // 2

        start_a_r, end_a_r = max(0, shift_r), min(H_a, shift_r + H_b)
        start_a_c, end_a_c = max(0, shift_c), min(W_a, shift_c + W_b)
        if H_b>2*H_a:
            start_b_r = max(0, shifts[i][1] - (end_a_r - start_a_r))  
            start_b_c = max(0, shifts[i][0] - (end_a_c - start_a_c))
        else:
            start_b_r, start_b_c = max(0, -shift_r), max(0, -shift_c)
        end_b_r, end_b_c = start_b_r + (end_a_r - start_a_r), start_b_c + (end_a_c - start_a_c)

        thread_id = numba.get_thread_id()
        temp_arrays[thread_id, start_a_r:end_a_r, start_a_c:end_a_c] += b[start_b_r:end_b_r, start_b_c:end_b_c]

    for i in range(num_threads):
        heightmap += temp_arrays[i]

    return heightmap


class LTGen:
    # all count / density tuning is defined for this map size and scaled by area
    REFERENCE_SIZE = 1024

    def __init__(self, size=1024):
        self.crater_dict = {}
        self.max_crater_diam = 8000
        self.max_dict_D = None      # largest diameter (m) present in crater_dict
        self.dict_map_size = None   # map size the crater profiles were built for
        self.set_map_size(size)

    def set_map_size(self, size):
        """
        Set the map size and rescale every area dependent quantity with it.
        All tuning constants are expressed per REFERENCE_SIZE**2 pixels, so
        `crater_num_scaler` converts them to the current map area.
        """
        self.map_size = size
        self.crater_num_scaler = (size ** 2) / (self.REFERENCE_SIZE ** 2)
        if self.crater_dict and self.dict_map_size != size:
            print(f"WARNING: crater profiles were precomputed for map size {self.dict_map_size}. "
                  f"Please re-run `generate_crater_dict()` for map size {size}.")

    def _stochastic_int(self, x):
        """
        Round a (possibly fractional) count to an integer without biasing its
        expectation, e.g. 0.4 -> 0 60% of the time and 1 40% of the time.
        Truncating instead would silently drop features on small maps, where
        area scaled counts are routinely < 1.
        """
        x = max(0.0, float(x))
        n = int(x)
        return n + int(np.random.random() < (x - n))

    def _scaled_count(self, low, high):
        """
        Draw a feature count for the current map size. `low`/`high` are the
        counts tuned for a REFERENCE_SIZE map (same meaning as the bounds of
        np.random.randint); the draw is scaled by map area and stays valid when
        the scaled bounds collapse below 1.
        """
        lo = low * self.crater_num_scaler
        hi = high * self.crater_num_scaler
        if hi <= lo:
            return self._stochastic_int(lo)
        return self._stochastic_int(np.random.uniform(lo, hi))

    def generate_crater_dict(self, min_radius=5, max_radius=4000, low_memory=True):
        """Precompute crater profiles for efficiency during terrain generation."""
        if low_memory:
            self._precompute_crater_profiles(min_radius, max_radius, memory_saver=True)
        else:
            self._precompute_crater_profiles(min_radius, max_radius, memory_saver=False)

    def gen_highland(self, tot_age=4, k=10, dt=1/40, print_time=False, add_large_crats=True):
        """Generate a simulated heightmap of lunar highland terrain."""
        return self._generate_terrain(
            tot_age, k, dt, print_time, terr_type=0, add_large_crats=add_large_crats
        )

    def gen_maria(self, tot_age=3.6, k=25, dt=1/100, print_time=False, add_large_crats=True):
        """Generate a simulated heightmap of lunar maria terrain."""
        return self._generate_terrain(
            tot_age, k, dt, print_time, terr_type=1, add_large_crats=add_large_crats
        )

    def _crater_dict_ready(self):
        """Check that crater profiles exist and match the current map size."""
        if not self.crater_dict:
            print("Crater Dict has not been computed yet. Please run `generate_crater_dict()`.")
            return False
        if self.dict_map_size != self.map_size:
            print(f"WARNING: crater profiles were precomputed for map size {self.dict_map_size}, "
                  f"but the map size is {self.map_size}. Re-run `generate_crater_dict()` for "
                  f"consistent results.")
        return True

    def generate_non_eroded_terrain(self, age=1):
        """Generate terrain with precomputed craters but no erosion."""
        if not self._crater_dict_ready():
            return None
        print(f"GENERATING NON-ERODED TERRAIN OF AGE: {age}")
        heightmap = np.zeros((self.map_size, self.map_size))
        return self.apply_precomputed_craters(heightmap, self.crater_dict, age)

    def _generate_terrain(self, tot_age, k, dt, print_time, terr_type, add_large_crats):
        if not self._crater_dict_ready():
            return None

        terrain_name = "HIGHLAND" if terr_type == 0 else "MARIA"
        print(f"GENERATING {terrain_name} TERRAIN")

        ages = np.arange(tot_age * 10 - 1)[::-1] / 10 + 0.2
        hm = np.zeros((self.map_size, self.map_size))
        start = time.time()

        for age in ages:
            hm = self.apply_precomputed_craters_range(hm, self.crater_dict, age=age, terr_type=terr_type, add_large_crats=add_large_crats)
            hm = self.diffuse_hm_np(hm, 0.1, k=k, dt=dt)

        hm = self.apply_precomputed_craters(hm, self.crater_dict, age=0.1, terr_type=terr_type) + self.get_noise(terr_type=terr_type)
        if print_time:
            print(f"Total time: {time.time() - start:.2f} seconds")

        final_dt = 0.1 if terr_type == 0 else 0.03
        return self.diffuse_hm_np(hm, final_dt, k=k, dt=dt)

    def _precompute_crater_profiles(self, min_radius, max_radius, memory_saver=False):
        """Internal method to precompute crater profiles, optionally in memory-saving mode."""
        print("PRECOMPUTING CRATER PROFILES" + (" (MEMORY SAVER)" if memory_saver else ""))
        size = 2 * self.map_size
        maxY, maxX = np.mgrid[-size//2:size//2, -size//2:size//2]
        max_D = max_radius * 2

        def compute_profile(D):
            scaled_size = min(D * 3, size)
            if D > size:
                scaled_size = D
            y, x = (maxY, maxX) if scaled_size == size else np.mgrid[-scaled_size//2:scaled_size//2, -scaled_size//2:scaled_size//2]
            r = np.sqrt(x ** 2 + y ** 2)
            R = D / 2
            c4 = (-8.68e-5) * (D ** 2) - 0.131 * D
            norm_c = c4 * (-8) / (D * (np.log10(D) ** 2) / 2)
            c = (-D / 8) * ((np.log10(D) ** 2) / 2)
            coeffs = [
                ((-7.86e-5) * (D ** 2) - 0.129 * D) / norm_c,
                ((-1.21e-4) * (D ** 2) - 0.075 * D) / norm_c,
                ((4.44e-4) * (D ** 2) + 0.554 * D) / norm_c,
                ((-2.44e-4) * (D ** 2) - 0.350 * D) / norm_c
            ]
            h_r = 0.02513 * D
            alpha = 3.6
            crater_profile = np.zeros_like(r)
            crater_profile[r / R < 0.1] = c + h_r
            mask = (r / R >= 0.1) & (r / R <= 1)
            crater_profile[mask] = (
                coeffs[3] * (r[mask] / R) ** 3 +
                coeffs[2] * (r[mask] / R) ** 2 +
                coeffs[1] * (r[mask] / R) +
                coeffs[0] + h_r
            )
            crater_profile[r / R > 1] = h_r * np.exp(-alpha * ((r[r / R > 1] / R) - 1))
            crater_profile -= crater_profile[r / R > 1].min()
            return crater_profile

        stored_D = None
        for D in tqdm(range(min_radius * 2, max_D)):
            if memory_saver:
                if D > 5000:
                    ref_D = (D // 500) * 500
                elif D > 4000:
                    ref_D = (D // 400) * 400
                elif D > 3000:
                    ref_D = (D // 300) * 300
                elif D > 2000:
                    ref_D = (D // 200) * 200
                elif D > 1000:
                    ref_D = (D // 100) * 100
                elif D > 500:
                    ref_D = (D // 50) * 50
                elif D > 200:
                    ref_D = (D // 20) * 20
                else:
                    ref_D = D
            else:
                if D > 5000:
                    ref_D = (D // 500) * 500
                elif D > 4000:
                    ref_D = (D // 200) * 200
                elif D > 3000:
                    ref_D = (D // 100) * 100
                elif D > 2000:
                    ref_D = (D // 50) * 50
                elif D > 1000:
                    ref_D = (D // 20) * 20
                elif D > 500:
                    ref_D = (D // 10) * 10
                elif D > 200:
                    ref_D = (D // 5) * 5
                else:
                    ref_D = D

            if stored_D != ref_D:
                self.crater_dict[ref_D] = compute_profile(ref_D)
                stored_D = ref_D
            self.crater_dict[D] = self.crater_dict[ref_D]

        self.max_dict_D = max_D - 1
        self.dict_map_size = self.map_size

        return self.crater_dict, max_D - 1

    
    def NcumDover1km(self, t):
        """
            get cumulative number of craters over 1km for a given age
            t: age in Gyr
        """
        return (5.44e-14)*(np.exp(6.93*t)-1)+(8.38e-4)*t


    def neukum_prod_new(self, D, a0=-3.087):
        """
            neukum crater diameter production function with new 
            parameters
        """
        a = np.array([a0, -3.557528, 0.781027, 1.021521, -0.156012, -0.444058,
                        0.019977, 0.086850, -0.005874, -0.006809, 8.25e-4, 5.54e-5])
        j = np.arange(12)
        logD = np.log10(D)
        sum = np.sum(a*np.power(logD, j))
        return 10**(sum)


    def _clip_diams(self, Di):
        """
        Keep sampled diameters (km) inside the range covered by crater_dict so
        the lookup in the apply_* methods can never miss.
        """
        upper = self.max_crater_diam*1000 - 1
        if self.max_dict_D is not None:
            upper = min(upper, self.max_dict_D / 1000)
        return np.clip(Di, 0.01, upper)

    def get_crat_diam(self, D=0.01, age=3.5):
        """
            Get random sample of crater diameters of a
            1km surface of a given age
            D: crater diameter in km
            age: age in Gyr

            The size-frequency distribution is a property of the surface, not of
            the tile we are rendering, so `crater_num_scaler` only scales how
            many craters are drawn; the diameters themselves stay normalized to
            the reference production function.
        """
        ncumDo10 = self.NcumDover1km(age)
        a0 = np.log10(ncumDo10)
        # print(a0)
        b=3.35
        ncumD = self.neukum_prod_new(D, a0=a0)
        num_crats = self._stochastic_int(ncumD*self.crater_num_scaler)
        ri = np.random.uniform(0, 1, num_crats)
        Di = self._clip_diams(np.power((ri*ncumD)/ncumDo10, -1/b))
        return Di

    # will give you a list of D's based on an age and a time range
    def get_crat_diam2(self, D=0.01, age=3.5, step=0.5):
        """
            Get random sample of crater diameters of a 
            1km surface of a given age, for a given time range
            D: crater diameter in km
            age: age in Gyr
        """
        ncumDo10 = self.NcumDover1km(age)
        a0 = np.log10(ncumDo10)
        b=3.35

        ncumD1 = self.neukum_prod_new(D, a0=a0)
        ncumDo101 = self.NcumDover1km(age-step)
        a01 = np.log10(ncumDo101)
        b=3.35
        ncumD2 = self.neukum_prod_new(D, a0=a01)
        ncumD = ncumD1-ncumD2

        # as in get_crat_diam, map area only changes the crater count, while the
        # diameters stay normalized to the reference production function
        num_crats = self._stochastic_int(ncumD*self.crater_num_scaler)
        ri = np.random.uniform(0, 1, num_crats)
        Di = self._clip_diams(np.power((ri*ncumD)/(ncumDo10-ncumDo101), -1/b))
        return Di

    def NoverD(self, D, t):
        return 5.44 * 10e-14*[np.exp(6.93*t)-1] + 8.38 * 10e-4*t
    
    def shift_and_add(self, a: np.ndarray, b: np.ndarray, center_shift: tuple):
        """
        Adds `b` to `a` while keeping `a`'s shape, shifting `b` such that its center aligns with `center_shift` in `a`.
        Any part of `b` outside `a` is ignored.

        Parameters:
            a (np.ndarray): Base array with shape (H, W).
            b (np.ndarray): Larger array to be added after shifting, with shape (H_b, W_b).
            center_shift (tuple): The (row, col) in `a` where the center of `b` should align.

        Returns:
            np.ndarray: The modified `a` array after addition.
        """
        H_a, W_a = a.shape
        H_b, W_b = b.shape

        shift_r, shift_c = center_shift[0] - H_b // 2, center_shift[1] - W_b // 2
        start_a_r, end_a_r = max(0, shift_r), min(H_a, shift_r + H_b)
        start_a_c, end_a_c = max(0, shift_c), min(W_a, shift_c + W_b)
        start_b_r, start_b_c = max(0, -shift_r), max(0, -shift_c)
        end_b_r, end_b_c = start_b_r + (end_a_r - start_a_r), start_b_c + (end_a_c - start_a_c)

        if not a.flags['C_CONTIGUOUS']:
            a = np.ascontiguousarray(a)

        np.add(a[start_a_r:end_a_r, start_a_c:end_a_c],
            b[start_b_r:end_b_r, start_b_c:end_b_c],
            out=a[start_a_r:end_a_r, start_a_c:end_a_c])
        return a


    def apply_precomputed_craters(self, heightmap, crater_dict, age=3.5, terr_type=0):
        """
        generates a cratered terrain given surface age
        heightmap: np.ndarray to add craters to
        crater_dict: a dict of crater profiles based on diameter
        """
        rows, cols = heightmap.shape
        craters = []
        centers = []
        shifts = []
        # D = 1000*sample_crater_diameters(crater_num, DS, D_PROBS)
        D = self.get_crat_diam(age = age)*1000
        for i in range(len(D)):
            cx = np.random.randint(0, cols)
            cy = np.random.randint(0, rows)
            centers.append((cx, cy))

            # compute crater size
            if terr_type ==0:
                crat_scalar = (np.random.exponential(1)/20)
            else:
                crat_scalar =  np.min((np.random.exponential(1)/3, 1.8))/np.log10(D[i])
                
            crater_profile = crater_dict[int(D[i])]*crat_scalar
            crat_row, crat_col = crater_profile.shape
            shiftx = np.random.randint(0, crat_col)
            shifty = np.random.randint(0, crat_row)
            shifts.append((shiftx, shifty))
            craters.append(crater_profile)

        # numba cannot infer the element type of an empty list, and small maps
        # legitimately draw zero craters for a given age slice
        if not craters:
            return heightmap

        # apply all craters in a single batch
        return batch_shift_and_add_numba(heightmap, craters, centers, shifts)


    def apply_precomputed_craters_range(self, heightmap, crater_dict, age=3.5, terr_type=0, add_large_crats=True):
        """
        generates a cratered terrain given surface age, for a
        heightmap: np.ndarray to add craters to
        crater_dict: a dict of crater profiles based on diamt
        type: 0 for highland, 1 for maria
        """
        rows, cols = heightmap.shape
        craters = []
        centers = []
        shifts = []

        if terr_type==0:
            large_crat_cuttoff = .9685
        else:
            large_crat_cuttoff = .9963

        D = self.get_crat_diam2(age = age, step=.1)*1000
        if (age -.1) > 2.6:
            D = D[D>20]
            
        # keep the basin-scale draw inside the precomputed profile range instead
        # of clamping to it, which would pile every oversized draw onto one size
        large_D_max = 7800 if self.max_dict_D is None else min(7800, self.max_dict_D + 1)
        if add_large_crats and large_D_max > 500:
            large_D = np.random.randint(500, large_D_max)
            # per-step chance of dropping in a basin-scale crater. Such a crater
            # only has to *overlap* the map to shape it, so the odds of one being
            # present scale with (map + D)**2 rather than with map area: a 7.8 km
            # crater is nearly as likely to touch a 256 tile as a 1024 one, while
            # a 500 m crater is not. Equals 1 at the reference size.
            reach_scaler = ((self.map_size + large_D) ** 2 /
                            (self.REFERENCE_SIZE + large_D) ** 2)
            large_crat_prob = (1 - np.power(large_crat_cuttoff, age)) * reach_scaler
            if np.random.rand() < large_crat_prob:
                D= np.concatenate(([large_D], D))

        for i in range(len(D)):
            cx = np.random.randint(0, cols)
            cy = np.random.randint(0, rows)
            centers.append((cx, cy))

            # compute crater size
            if terr_type ==0:
                crat_scalar = (np.random.exponential(1)/20)
            else:
                crat_scalar =  np.min((np.random.exponential(1)/3, 1.8))/np.log10(D[i])
                
            crater_profile = crater_dict[int(D[i])] * crat_scalar
            crat_row, crat_col = crater_profile.shape
            shiftx = np.random.randint(0, crat_col)
            shifty = np.random.randint(0, crat_row)
            shifts.append((shiftx, shifty))
            craters.append(crater_profile)

        if not craters:
            return heightmap

        # apply all craters in a single batch
        # print()
        return batch_shift_and_add_numba(heightmap, craters, centers, shifts)

    def add_rocks(self, heightmap, num_rocks=80, size_range=(1, 3), height_range=(0.5, 2)):
        """
        adds random polygons to a heightmap to imitate rocks
        """
        size = heightmap.shape[0]

        for _ in range(num_rocks):
            cx, cy = np.random.randint(0, size, 2)

            num_points = np.random.randint(3, 7)  # Triangles to hexagons
            angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
            radii = np.random.uniform(size_range[0], size_range[1], num_points)

            x_poly = (radii * np.cos(angles) + cx).astype(int)
            y_poly = (radii * np.sin(angles) + cy).astype(int)

            points = np.column_stack([x_poly, y_poly])
            points = np.unique(points, axis=0)

            if len(points) < 3:
                continue

            try:
                hull = ConvexHull(points)
            except:
                continue  # skip this iteration if ConvexHull fails

            rr, cc = polygon(points[hull.vertices, 1], points[hull.vertices, 0], heightmap.shape)
            heightmap[rr, cc] += np.random.uniform(height_range[0], height_range[1])  # Random rock height

        return heightmap

    def get_noise(self, terr_type=0):
        """
        add high freq perlin noise, rocks, and small craters
        
        terr_type: 0 for highland, 1 for maria
        """
        size = self.map_size
        scale = 24
        # the noise is defined in pixel units, so the octave count must not drop
        # below the reference tuning on smaller maps or they come out smoother
        octaves = 5 + max(0, int(np.log2(np.sqrt(self.crater_num_scaler))))
        persistence = 0.5
        lacunarity = 2.0

        noise_rocks = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                noise_rocks[i][j] = pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=size,
                                        repeaty=size,
                                        base=42)
                
        # rock counts are per reference tile and get scaled by area in
        # _scaled_count; the size/height ranges are in pixels and stay fixed
        if terr_type ==0:
            pos_rock_num = (20, 150)
            pos_rock_range = (1,9)
            pos_rock_height = (.1, 2)
            neg_rock_num = (1, 20)
            neg_rock_range = (1,7)
            neg_rock_height = (-2, -.1)
        else:
            pos_rock_num = (1, 15)
            pos_rock_range = (1,7)
            pos_rock_height = (.1, 1)
            neg_rock_num = (1, 3)
            neg_rock_range = (1,5)
            neg_rock_height = (-1, -.1)


        noise_rocks = (noise_rocks - np.min(noise_rocks)) / (np.max(noise_rocks) - np.min(noise_rocks)) * 1.5
        noise_rocks = self.add_rocks(noise_rocks, self._scaled_count(*pos_rock_num), size_range=pos_rock_range, height_range=pos_rock_height)
        noise_rocks = self.add_rocks(noise_rocks, self._scaled_count(*neg_rock_num), size_range=neg_rock_range, height_range=neg_rock_height)

        num_craters = self._scaled_count(5, 80)
        for _ in range(num_craters):
            cx, cy = np.random.randint(0, size, 2)
            crater_radius = np.random.uniform(1, 7)
            y, x = np.ogrid[:size, :size]
            mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * crater_radius ** 2))
            noise_rocks -= mask * np.random.uniform(0.1, 2)  
        return noise_rocks

    def laplacian_2d_numpy(self, u, h):
        """
        compute the laplacian for a 2d heightmap
        """
        return (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u) / h**2


    def diffuse_hm_np(self, u, Gyr, k=10, dt=1/40):
        """
        simulate topographic diffusion for a cratered surface
        u: np.ndarray: heightmap to diffused
        Gyr: Giga year time to diffuse
        k: diffusion coefficient
        dt: time step in Gyr
        """
        pad_width =10
        u_padded = np.pad(u, pad_width=pad_width, mode='reflect') 

        for i in range(int(Gyr/dt)):
            u_padded += k * dt * self.laplacian_2d_numpy(u_padded, 1)

        u_result = u_padded[pad_width:-pad_width, pad_width:-pad_width]  # Remove the 1-cell padding from all sides
        return u_result
    
    def display_DTM(self, hm, title='heightmap', vmin=None, vmax=None, show=True):
        plt.clf()
        plt.figure(figsize=(8, 8))
        img = plt.imshow(hm, cmap="gray", vmin=vmin, vmax=vmax)  # Set vmin and vmax for scaling
        plt.colorbar(img, label="Elevation (m)")  # Color bar reflects the set min/max
        plt.title(f"{title}")
        if show:
            plt.show()
        return plt

    
    
    
    
if __name__ == "__main__":
    mygenerator = LTGen()

    print("attempting terrain gen before dict created")
    sample_terr = mygenerator.generate_non_eroded_terrain()
    highlands = mygenerator.gen_highland()
    mygenerator.generate_crater_dict(low_memory=True)
    sample_terr = mygenerator.generate_non_eroded_terrain()
    mygenerator.display_DTM(sample_terr, "non eroded terr of age 1")
    highlands = mygenerator.gen_highland()
    mygenerator.display_DTM(highlands, "highland terrain")
    