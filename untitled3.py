import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev

# -------------------------
# 1) Input Data
# -------------------------
raw_data = np.array([
    [-5500, 0, 8000], [-5500, 4905, 10400], [-5500, 7010, 12800],
    [0, 0, 6588], [0, 2624, 8000], [0, 6507, 10400], [0, 8343, 12800],
    [1641, 0, 2000], [1777, 0, 6000], [2984, 0, 4000], [5339, 0, 0],
    [13500, 329, 0], [13500, 1077, 2000], [13500, 2918, 4000], [13500, 5137, 6000],
    [13500, 7481, 8000], [13500, 9627, 10400], [13500, 10690, 12800],
    [27000, 353, 0], [27000, 4882, 2000], [27000, 7242, 4000], [27000, 8986, 6000],
    [27000, 10210, 8000], [27000, 11080, 10400], [27000, 11440, 12800],
    [40500, 2195, 0], [40500, 8260, 2000], [40500, 10140, 4000], [40500, 10960, 6000],
    [40500, 11360, 8000], [40500, 11490, 10400], [40500, 11500, 12800],
    [54000, 5865, 0], [54000, 10460, 2000], [54000, 11350, 4000], [54000, 11500, 6000],
    [54000, 11500, 8000], [54000, 11500, 10400], [54000, 11500, 12800],
    [67500, 8500, 0], [67500, 11330, 2000], [67500, 11500, 4000], [67500, 11500, 6000],
    [67500, 11500, 8000], [67500, 11500, 10400], [67500, 11500, 12800],
    [81000, 6391, 0], [81000, 10080, 2000], [81000, 10930, 4000], [81000, 11270, 6000],
    [81000, 11390, 8000], [81000, 11480, 10400], [81000, 11500, 12800],
    [94500, 3314, 0], [94500, 7551, 2000], [94500, 8765, 4000], [94500, 9453, 6000],
    [94500, 9866, 8000], [94500, 10310, 10400], [94500, 10820, 12800],
    [108000, 986, 0], [108000, 4452, 2000], [108000, 5618, 4000], [108000, 6374, 6000],
    [108000, 6990, 8000], [108000, 7803, 10400], [108000, 8724, 12800],
    [121500, 205, 0], [121500, 1577, 2000], [121500, 2251, 4000], [121500, 2809, 6000],
    [121500, 3360, 8000], [121500, 4247, 10400], [121500, 5479, 12800], [121500, 6999, 15200],
    [130300, 0, 0], [133200, 0, 2000], [133900, 0, 4000], [134500, 0, 6000],
    [135000, 0, 8000], [135000, 0, 8001], [135000, 411, 10400], [135000, 1191, 12800],
    [135000, 2382, 15200], [135700, 0, 10400], [136600, 0, 12800], [137600, 0, 15200]
])

# Mirror across centerline
positive_side = raw_data[raw_data[:, 1] > 0].copy()
mirrored = positive_side.copy()
mirrored[:, 1] *= -1
points = np.vstack([raw_data, mirrored])

# Ensure unique points to avoid duplicate vertex issues
points = np.unique(points, axis=0)

# -------------------------
# 2) Process Slices & Smooth
# -------------------------
# Parameters
POINTS_PER_SLICE = 50   # Resolution of the curve (higher = smoother)
SMOOTH_FACTOR = 1.0     # Spline smoothness (0 = loose, higher = tighter)

# Identify unique X-stations (slices)
unique_x = np.sort(np.unique(points[:, 0]))

X_list = []
Y_list = []
Z_list = []

for x_val in unique_x:
    # Get all points at this X station
    # We use a small tolerance because floats might not be exact
    mask = np.abs(points[:, 0] - x_val) < 1.0
    slice_pts = points[mask]

    # Skip slices with too few points to form a curve
    if len(slice_pts) < 3:
        continue

    # --- Sorting Logic ---
    # We want to draw a line from Port -> Keel -> Starboard.
    # Sorting by Y (beam) works perfectly for a U-shaped hull.
    # (Leftmost Y is port, Rightmost Y is starboard).
    sort_idx = np.argsort(slice_pts[:, 1])
    sorted_pts = slice_pts[sort_idx]
    
    y_slice = sorted_pts[:, 1]
    z_slice = sorted_pts[:, 2]

    # --- Spline Smoothing ---
    try:
        # 'splprep' fits a B-spline representation of the curve
        # s=0 would force it through every point (jagged), s>0 smooths it.
        # k=3 is a cubic spline (curved), k=1 is linear (straight lines).
        tck, u = splprep([y_slice, z_slice], s=0, k=min(3, len(slice_pts)-1))
        
        # Evaluate the spline at equally spaced points
        u_new = np.linspace(0, 1, POINTS_PER_SLICE)
        new_y, new_z = splev(u_new, tck)
        
        X_list.append(np.full(POINTS_PER_SLICE, x_val))
        Y_list.append(new_y)
        Z_list.append(new_z)
        
    except Exception as e:
        print(f"Skipping slice at X={x_val}: {e}")
        continue

# Convert lists to 2D grids (Meshgrid format)
X_mesh = np.array(X_list)
Y_mesh = np.array(Y_list)
Z_mesh = np.array(Z_list)

# -------------------------
# 3) Plot
# -------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot raw points for reference (optional)
ax.scatter(points[:,0], points[:,1], points[:,2], c='k', s=5, alpha=0.3, label='Raw Data')

# Plot Surface
# rstride/cstride controls how dense the wireframe grid is drawn
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, 
                       cmap='Blues',      # Blueish color map
                       alpha=0.8,         # Slightly transparent
                       edgecolor='none',  # No wireframe lines for maximum smoothness
                       rcount=50, ccount=50) # render resolution

# Add simple lighting visual cues
# (Matplotlib lighting is basic, but this helps 3D perception)
ax.set_title('Reconstructed Ship Hull (Smoothed)')
ax.set_xlabel('X (Length)')
ax.set_ylabel('Y (Beam)')
ax.set_zlabel('Z (Height)')

# Set equal aspect ratio implies a realistic view
# (Matplotlib 3D doesn't handle aspect ratio perfectly, but this helps)
max_range = np.array([X_mesh.max()-X_mesh.min(), Y_mesh.max()-Y_mesh.min(), Z_mesh.max()-Z_mesh.min()]).max() / 2.0
mid_x = (X_mesh.max()+X_mesh.min()) * 0.5
mid_y = (Y_mesh.max()+Y_mesh.min()) * 0.5
mid_z = (Z_mesh.max()+Z_mesh.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Adjust viewing angle
ax.view_init(elev=25, azim=-135)

plt.tight_layout()
plt.show()