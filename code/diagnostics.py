
import numpy as np

def calculate_symmetry_index(zb, N):
    """
    Calculates the symmetry of the bed profile to classify bar types.
    
    Args:
        zb (array): Bed elevation vector (Chebyshev spectral space or grid space).
                    If input is spectral, it should be converted to grid space first.
        N (int): Number of Chebyshev points (grid size - 1).

    Returns:
        float: Symmetry Index (0 = Perfectly Symmetric/Central, 1 = Anti-symmetric/Alternate).
        normalized_diff: Raw difference metric.
    """
    # Assuming zb is already in physical space (grid points from y=1 to y=-1)
    # If not, strictly the caller should handle transform, but for robust design:
    if len(zb) != N + 1:
        # Fallback or error, but here assuming correct length commensurate with N
        pass

    # Use Real part (or complex) relative difference
    # Odd Symmetry (Alternate): Left ~ -Right
    # Even Symmetry (Central): Left ~ Right
    
    mid = N // 2
    # e.g., N=16, 0..8..16. Mid point 8.
    # Left: 0..7. Right Reversed: 16..9.
    
    val_left = zb[:mid]
    val_right_reversed = zb[-(mid):][::-1]
    
    # Check Even-ness: |L - R| / |L|
    # If perfect even, diff is 0.
    diff_even = np.mean(np.abs(val_left - val_right_reversed))
    
    # Check Odd-ness: |L + R| / |L|
    # If perfect odd, L + R = L + (-L) = 0.
    diff_odd = np.mean(np.abs(val_left + val_right_reversed))
    
    norm = np.mean(np.abs(val_left)) + 1e-10
    
    score_even = diff_even / norm
    
    # We return score_even as "Symmetry Index"
    # S ~ 0 => Central (Even)
    # S ~ 2 => Alternate (Odd)
    return score_even

def find_bar_axis(zb, y_grid):
    """
    Finds the transverse location of the maximum deposition (bar peak).
    
    Args:
        zb (array): Bed elevation.
        y_grid (array): Transverse coordinates (zeta).
        
    Returns:
        float: Zeta coordinate of the peak.
    """
    idx_max = np.argmax(np.real(zb)) # Use real part or absolute? Typically deposition is real>0
    # For linear modes, phase is arbitrary, so we look at Magnitude envelope usually,
    # OR we look at the phase-shifted profile where max is positive.
    # For mode analysis, using Amplitude Envelope ensures robustness.
    return y_grid[np.argmax(np.abs(zb))]

def classify_mode(symmetry_index, growth_rate):
    """
    Classifies the mode based on diagnostics.
    
    Returns:
        int: Mode Type (1=Central, 2=Alternate, 3=Wall/Complex, 0=Stable/Noise)
    """
    if growth_rate > 0.5:
        return 3 # Wall Mode likely (High Gr)
    
    if symmetry_index < 0.2:
        return 1 # Central Bar
    else:
        return 2 # Alternate Bar
