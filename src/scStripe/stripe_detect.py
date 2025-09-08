#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
import ruptures as rpt
import math
import cooler
from statsmodels.stats.multitest import multipletests
import diptest


def make_symmetric(matrix):  
    rows = np.shape(matrix)[0]
    cols = np.shape(matrix[0])[0]
    for i in range(rows):
        for j in range(i+1, cols):
            matrix[i][j] = matrix[j][i]
    return matrix



def extract_submatrix_by_diag(matrix, k_start, k_end, c_start, c_end, side="left"):
    if c_start > c_end:
        raise ValueError("c_start must be <= c_end")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    ks = range(k_start, k_end + 1)
    diags = [np.diag(matrix, k=k) for k in ks]

    if side == "right":
        diags = [np.pad(vec, (k, 0), mode="constant") for vec, k in zip(diags, ks)]

    rows = [vec[c_start: c_end + 1] for vec in diags]
    return np.asarray(rows)



def compare_eigenvalues(nums):
    if np.all(nums == 0):
        return None
    
    for i in range(np.shape(nums)[0] - 1):
        if nums[i+1] == 0:
            return i
        if i > 0 and nums[i-1] / nums[i] < 1.1 and nums[i] / nums[i+1] < 1.1:
            return i
    return None




def get_stripe_algo_from_matrix(mat_observed, top_i, compare_eigenvalues_fn):
    mat_sym = (mat_observed + mat_observed.T) / 2
    eigenvalue_B, featurevector_B = np.linalg.eigh(mat_sym)
    sorted_indices = np.argsort(eigenvalue_B)[::-1]
    top_10_indices = sorted_indices[:10]

    if top_i == 0:
        n_selected = compare_eigenvalues_fn(eigenvalue_B[top_10_indices])
        featurevector_topi = featurevector_B[:, top_10_indices[:n_selected]]
    else:
        featurevector_topi = featurevector_B[:, top_10_indices[:top_i]]

    algo = rpt.Pelt(model="rbf", jump=2).fit(featurevector_topi)
    return algo




def stripe_t_test_len_slant_within_out(mat_observed_big,dry_stripe_width_loci,dry_stripe_length_loci,extend_bins, len_select_bins):
    if np.mean(dry_stripe_width_loci) < dry_stripe_length_loci:
        left_or_right = "left"
        stripe_width = [min(dry_stripe_width_loci),max(dry_stripe_width_loci)]
        stripe_length = dry_stripe_length_loci - max(dry_stripe_width_loci)

    ##### wid
        kk = max(dry_stripe_width_loci)-min(dry_stripe_width_loci)+1
        within_dry_stripe_wid = extract_submatrix_by_diag(mat_observed_big, 1, stripe_length, stripe_width[0]+extend_bins, stripe_width[1]+extend_bins, "left")
        out_dry_stripe_wid_up = extract_submatrix_by_diag(mat_observed_big, 1, stripe_length, stripe_width[0]-kk+extend_bins, stripe_width[0]-1+extend_bins, "left")
        out_dry_stripe_wid_down = extract_submatrix_by_diag(mat_observed_big, 1, stripe_length, stripe_width[1]+1+extend_bins, stripe_width[1]+kk+extend_bins, "left")

    ##### len
        within_dry_stripe_len = extract_submatrix_by_diag(mat_observed_big, (stripe_length+stripe_width[1]-stripe_width[0])-len_select_bins+1, (stripe_length+stripe_width[1]-stripe_width[0]), stripe_width[0]+extend_bins, stripe_width[1]+extend_bins, "left")
        out_dry_stripe_len = extract_submatrix_by_diag(mat_observed_big, (stripe_length+stripe_width[1]-stripe_width[0])+1, (stripe_length+stripe_width[1]-stripe_width[0])+len_select_bins, stripe_width[0]+extend_bins, stripe_width[1]+extend_bins, "left")


    else: 
        left_or_right = "right"
        stripe_width = [min(dry_stripe_width_loci),max(dry_stripe_width_loci)]
        stripe_length = min(dry_stripe_width_loci)+1 - dry_stripe_length_loci

    ##### wid
        kk = max(dry_stripe_width_loci)-min(dry_stripe_width_loci)+1
        within_dry_stripe_wid = extract_submatrix_by_diag(mat_observed_big, 1, stripe_length, stripe_width[0]+extend_bins, stripe_width[1]+extend_bins, "right")
        out_dry_stripe_wid_up = extract_submatrix_by_diag(mat_observed_big, 1, stripe_length, stripe_width[0]-kk+extend_bins, stripe_width[0]-1+extend_bins, "right")
        out_dry_stripe_wid_down = extract_submatrix_by_diag(mat_observed_big, 1, stripe_length, stripe_width[1]+1+extend_bins, stripe_width[1]+kk+extend_bins, "right")

    ##### len
        within_dry_stripe_len = extract_submatrix_by_diag(mat_observed_big, (stripe_length+(stripe_width[1]-stripe_width[0]))-len_select_bins+1, (stripe_length+(stripe_width[1]-stripe_width[0])), stripe_width[0]+extend_bins, stripe_width[1]+extend_bins, "right")
        out_dry_stripe_len = extract_submatrix_by_diag(mat_observed_big, (stripe_length+(stripe_width[1]-stripe_width[0]))+1, (stripe_length+(stripe_width[1]-stripe_width[0]))+len_select_bins, stripe_width[0]+extend_bins, stripe_width[1]+extend_bins, "right")
    

    within_dry_stripe_wid_mean = np.mean(within_dry_stripe_wid, axis=1)
    out_dry_stripe_wid_up_mean = np.mean(out_dry_stripe_wid_up, axis=1)
    out_dry_stripe_wid_down_mean = np.mean(out_dry_stripe_wid_down, axis=1)

    within_dry_stripe_len_mean = np.mean(within_dry_stripe_len, axis=0)
    out_dry_stripe_len_mean = np.mean(out_dry_stripe_len, axis=0)

    foldchange_wid_up = np.mean(within_dry_stripe_wid_mean)/np.mean(out_dry_stripe_wid_up_mean)
    foldchange_wid_down = np.mean(within_dry_stripe_wid_mean)/np.mean(out_dry_stripe_wid_down_mean)
    foldchange_len = 4

    within_dry_stripe_wid_mean = np.where(within_dry_stripe_wid_mean <= 0, 1e-10, within_dry_stripe_wid_mean)
    out_dry_stripe_wid_up_mean = np.where(out_dry_stripe_wid_up_mean <= 0, 1e-10, out_dry_stripe_wid_up_mean)
    out_dry_stripe_wid_down_mean = np.where(out_dry_stripe_wid_down_mean <= 0, 1e-10, out_dry_stripe_wid_down_mean)
    within_dry_stripe_len_mean = np.where(within_dry_stripe_len_mean <= 0, 1e-10, within_dry_stripe_len_mean)
    out_dry_stripe_len_mean = np.where(out_dry_stripe_len_mean <= 0, 1e-10, out_dry_stripe_len_mean)

    _, dip_p_value = diptest.diptest(np.log(within_dry_stripe_wid_mean))

    # t test
    wid_up = stats.ttest_rel(np.log(within_dry_stripe_wid_mean), np.log(out_dry_stripe_wid_up_mean))
    p_wid_up = wid_up.pvalue
    wid_up_single_tail = p_wid_up / 2

    wid_down = stats.ttest_rel(np.log(within_dry_stripe_wid_mean), np.log(out_dry_stripe_wid_down_mean))
    p_wid_down = wid_down.pvalue
    wid_down_single_tail = p_wid_down / 2

    len = stats.ttest_rel(np.log(within_dry_stripe_len_mean), np.log(out_dry_stripe_len_mean))
    p_len = len.pvalue
    len_single_tail = p_len / 2


    if np.isnan(p_wid_up) or np.isnan(p_wid_down) or np.isnan(p_len):
        print("Skipping this test due to invalid p-values.")
        return None, None, None, None, None, None, None, None, None, None
    else:
        return stripe_width, stripe_length, left_or_right, wid_up_single_tail, wid_down_single_tail, len_single_tail, foldchange_wid_up, foldchange_wid_down, foldchange_len, dip_p_value




def intervals_limit(data, max_width, min_length):
    filtered_intervals = []

    for row in data:
        start, end, anchor = row[:2][0], row[:2][1], row[2]
        width = end - start

        dist_to_edge = min(abs(anchor - start), abs(anchor - end))

        if width <= max_width and dist_to_edge >= min_length:
            filtered_intervals.append([min(start, end), max(start, end), anchor])

    return filtered_intervals



def calculate_observed_expected_matrix(matrix):
    oe_matrix = np.zeros_like(matrix, dtype=float)
    num_rows, num_cols = matrix.shape
    
    for i in range(num_rows):
        for j in range(num_cols):
            diagonal_elements = [matrix[k, k + (j - i)] for k in range(max(0, i - j), min(num_rows, num_cols - (j - i)))]
            diagonal_mean = np.mean(diagonal_elements)
            
            if diagonal_mean == 0:
                diagonal_mean = 1e-10  # 避免除零
            
            oe_matrix[i, j] = matrix[i, j] / diagonal_mean
    
    return oe_matrix



def get_candidate_stripe_foldchange(submat, t_row1, t_row2, t_col1, t_col2):
    region = submat[t_row1:t_row2, t_col1:t_col2]
    if region.size == 0:
        return 0
    
    region_mean = np.nanmean(region)
    background_mean = np.nanmean(submat)
    
    return region_mean / background_mean if background_mean else 0




def calculate_fold_changes(submat, fold_threshold, max_width, min_length, bkps, merge_len_kbins):
    def detect_stripe_end(from_idx, to_idx, axis_points, get_fc_func, is_forward=True):
        direction = range(len(axis_points)-1, -1, -1) if not is_forward else range(len(axis_points))
        peak_point = None
        peak_found = False

        for k in direction:
            if abs(axis_points[k] - to_idx if is_forward else from_idx - axis_points[k]) >= min_length:
                fc = get_fc_func(from_idx, to_idx, axis_points[k])
                if fc >= fold_threshold:
                    prev_fc = 0
                    loop_range = range(k, -1, -1) if not is_forward else range(k, len(axis_points))
                    for j in loop_range:
                        fc_next = get_fc_func(from_idx, to_idx, axis_points[j])
                        if fc_next < prev_fc and not peak_found:
                            peak_point = axis_points[j + 1] if not is_forward else axis_points[j - 1]
                            peak_found = True
                            break
                        prev_fc = fc_next
                    if peak_found:
                        break
        return peak_point

    def merge_candidates(candidates, use_max):
        merged = True
        while merged:
            merged = False
            result, used = [], set()
            for i, (a1, b1, d1) in enumerate(candidates):
                if i in used:
                    continue
                merged_once = False
                for j in range(i+1, len(candidates)):
                    if j in used:
                        continue
                    a2, b2, d2 = candidates[j]
                    if abs(min(a1, a2) - max(b1, b2)) > max_width:
                        continue
                    if (a1 == b2 or b1 == a2) and abs(d1 - d2) <= merge_len_kbins:
                        new_d = max(d1, d2) if use_max else min(d1, d2)
                        result.append((min(a1, a2), max(b1, b2), new_d))
                        used.update([i, j])
                        merged = True
                        merged_once = True
                        break
                if not merged_once:
                    result.append((a1, b1, d1))
            candidates = result
        return candidates

    def tune_stripe_length(stripes, axis_points, extract_func):
        tuned = []
        for a, b, d in stripes:
            candidates = [p for p in axis_points if extract_func(a, b, p)]
            if not candidates:
                tuned.append([a, b, d])
                continue
            scores = [
                np.nanmean(submat[a:b, a:col]) / np.nanmean(submat)
                if col > b else
                np.nanmean(submat[col:b, a:b]) / np.nanmean(submat)
                for col in candidates
            ]
            tuned.append([a, b, candidates[np.argmax(scores)]])
        return tuned

    results = set()
    for i in range(len(bkps) - 1):
        a, b = bkps[i], bkps[i + 1]
        if abs(b - a) > max_width:
            continue

        others = [p for j, p in enumerate(bkps) if j not in (i, i + 1)]
        right = sorted(p for p in others if p > b)
        left = sorted(p for p in others if p < a)

        # left stripe
        get_fc_h = lambda x1, x2, x3: get_candidate_stripe_foldchange(submat, x1, x2, x1, x3)
        right_end = detect_stripe_end(a, b, right, get_fc_h, is_forward=False)
        if right_end:
            results.add((a, b, right_end))

        # right stripe
        get_fc_v = lambda x1, x2, x3: get_candidate_stripe_foldchange(submat, x3, x2, x1, x2)
        left_end = detect_stripe_end(a, b, left, get_fc_v, is_forward=True)
        if left_end:
            results.add((a, b, left_end))

    res_list = list(results)
    gts = [r for r in res_list if r[2] > r[1]]
    lts = [r for r in res_list if r[2] <= r[1]]

    merged_gts = merge_candidates(gts, use_max=True)
    merged_lts = merge_candidates(lts, use_max=False)

    tuned_gts = tune_stripe_length(merged_gts, right, lambda a, b, p: p <= b and p - b >= min_length)
    tuned_lts = tune_stripe_length(merged_lts, left, lambda a, b, p: p >= a and a - p >= min_length)

    return tuned_gts + tuned_lts




def find_center_of_peak(avg_values, window_size=5, threshold=0.1):
    window_avg = np.convolve(avg_values, np.ones(window_size)/window_size, mode='valid')
    peaks = np.where(window_avg > np.max(window_avg) * threshold)[0]
    
    if len(peaks) == 0:
        return None
    
    center = peaks[np.argmax(window_avg[peaks])]

    return center + window_size // 2




def adjust_stripe_position(matrix, extend_bins, min_length, width_range, length_anchor, max_expand, change_threshold):

    start, end = width_range
    col_start = max(0, start - 5) + extend_bins
    col_end = min(matrix.shape[1], end + 5) + extend_bins

    if np.mean(width_range) < length_anchor:
        length = length_anchor - max(width_range)
        profile_matrix = extract_submatrix_by_diag(matrix, 1, length, col_start, col_end, "left")
    else:
        length = min(width_range) + 1 - length_anchor
        profile_matrix = extract_submatrix_by_diag(matrix, 1, length, col_start, col_end, "right")

    avg_profile = np.mean(profile_matrix, axis=0)
    peak_center = find_center_of_peak(avg_profile)

    if peak_center is None:
        return width_range, length_anchor

    left = right = peak_center
    while left > 0 and (right - left + 1) <= max_expand:
        change = abs(avg_profile[left] - avg_profile[left - 1]) / (avg_profile[left] + 1e-6)
        if change > change_threshold:
            break
        left -= 1

    while right < len(avg_profile) - 1 and (right - left + 1) <= max_expand:
        change = abs(avg_profile[right] - avg_profile[right + 1]) / (avg_profile[right] + 1e-6)
        if change > change_threshold:
            break
        right += 1

    if right - left + 1 > max_expand:
        half = max_expand // 2
        left = max(0, peak_center - half)
        right = min(len(avg_profile) - 1, peak_center + half)

    new_width_range = [left + col_start - extend_bins, right + col_start - extend_bins]

    if np.mean(new_width_range) < length_anchor:
        updated_length = length_anchor - max(new_width_range)
    else:
        updated_length = min(new_width_range) + 1 - length_anchor

    if updated_length < min_length:
        return width_range, length_anchor

    return new_width_range, length_anchor




def nosplit_and_call_stripes(matrix, top_k, penalty, fold_thresh, max_width, min_length):
    """
    Identify stripe regions from a Hi-C matrix without initial segmentation.

    Parameters:
    - matrix: np.ndarray, raw observed Hi-C matrix
    - top_k: int, number of top eigenvectors used in segmentation
    - penalty: float, penalty for changepoint detection
    - fold_thresh: float, fold change threshold for stripe detection
    - max_width: int, maximum allowed stripe width (in bins)
    - min_length: int, minimum allowed stripe length (in bins)

    Returns:
    - results: list of detected stripe information
    - bkps: list of changepoint positions
    """
    extend_bins = 50  # Number of bins to pad the matrix for boundary safety
    merge_len_bins = round(min_length / 2)
    len_select_bins = min_length

    mat_obs = matrix
    mat_oe = calculate_observed_expected_matrix(mat_obs)
    mat_n = mat_obs.shape[0]

    # === Construct background model ===
    mu = math.log(600) * (-0.7) + 1.0
    sigma = math.log(600) * (0.17) + 0.15
    mat_oe_big = np.random.normal(mu, sigma, (mat_n + 2 * extend_bins, mat_n + 2 * extend_bins))
    mat_oe_big = np.exp(mat_oe_big)
    mat_oe_big = make_symmetric(mat_oe_big)
    mat_oe_big[extend_bins:extend_bins + mat_n, extend_bins:extend_bins + mat_n] = mat_oe

    # === Changepoint detection using eigenvalue-based algorithm ===
    algo = get_stripe_algo_from_matrix(mat_obs, top_k, compare_eigenvalues)
    bkps = algo.predict(pen=penalty)[:-1] 

    if len(bkps) < 3:
        print("No stripe candidates found.")
        return [], bkps

    # === Detect stripe candidates based on fold change ===
    candidates = calculate_fold_changes(
        mat_oe, fold_thresh, max_width, min_length, bkps, merge_len_bins
    )
    valid_intervals = intervals_limit(candidates, max_width, min_length)

    # === Refine stripe width boundaries ===
    refined_candidates = []
    for width_loci, length_loci in [(w[:2], w[2]) for w in valid_intervals]:
        tuned_wid, tuned_len = adjust_stripe_position(
            mat_oe_big, extend_bins, min_length,
            width_loci, length_loci,
            max_width, change_threshold=0.5
        )
        refined_candidates.append([tuned_wid, tuned_len])

    # === Perform statistical testing on refined stripes ===
    results = []
    for width_loci, length_loci in refined_candidates:
        test_result = stripe_t_test_len_slant_within_out(
            mat_oe_big, width_loci, length_loci,
            extend_bins, len_select_bins
        )

        # Skip if p-values are invalid (e.g., empty region)
        if None in test_result[3:6]:
            print("Skipped stripe due to invalid p-values.")
            continue

        # Format: width range, length range, direction, p-values, fold changes, dip p-value
        result_entry = [
            test_result[0],                     # stripe width (start, end)
            [length_loci, test_result[1]],      # stripe length (anchor, length)
            test_result[2],                     # stripe direction ("left" or "right")
            *test_result[3:9]                   # p-values, fold changes, dip test p
        ]
        results.append(result_entry)

    return results, bkps




def split_and_call_stripes(matrix, top_k, penalty, fold_thresh, split_len, step_size, max_width, min_length):
    """
    Identify stripes in a large Hi-C matrix by splitting into submatrices.

    Parameters:
    - matrix: np.ndarray, full Hi-C observed matrix
    - top_k: int, number of top eigenvectors used in segmentation
    - penalty: float, penalty for changepoint detection
    - fold_thresh: float, fold change threshold
    - split_len: int, size of each submatrix
    - step_size: int, sliding step for submatrix split
    - max_width: int, maximum stripe width
    - min_length: int, minimum stripe length

    Returns:
    - stripe_results: list of detected stripes with statistics
    - all_breakpoints: list of all changepoint positions in global coordinates
    """
    extend_bins = 50
    merge_len_bins = round(min_length / 2)
    len_select_bins = min_length
    matrix_size = matrix.shape[0]

    mu = math.log(600) * (-0.7) + 1.0
    sigma = math.log(600) * 0.17 + 0.15

    sub_matrices = []
    sub_matrices_oe_big = []
    stripe_results = []
    all_changepoints = []

    # === Split matrix into submatrices ===
    if matrix_size < split_len:
        sub_matrices = [matrix]
        oe_big = np.random.normal(mu, sigma, (matrix_size + 2 * extend_bins, matrix_size + 2 * extend_bins))
        oe_big = np.exp(oe_big)
        oe_big = make_symmetric(oe_big)
        oe_big[extend_bins:extend_bins + matrix_size, extend_bins:extend_bins + matrix_size] = calculate_observed_expected_matrix(matrix)
        sub_matrices_oe_big.append(oe_big)
    else:
        for i in range(0, matrix_size, step_size):
            end = min(matrix_size, i + split_len)
            sub = matrix[i:end, i:end]
            sub_matrices.append(sub)

            if i - extend_bins >= 0 and i + split_len + extend_bins <= matrix_size:
                sub_full = matrix[i - extend_bins:i + split_len + extend_bins, i - extend_bins:i + split_len + extend_bins]
                oe_big = calculate_observed_expected_matrix(sub_full)
            else:
                oe_big = np.random.normal(mu, sigma, (sub.shape[0] + 2 * extend_bins, sub.shape[0] + 2 * extend_bins))
                oe_big = np.exp(oe_big)
                oe_big = make_symmetric(oe_big)
                oe_big[extend_bins:extend_bins + sub.shape[0], extend_bins:extend_bins + sub.shape[0]] = calculate_observed_expected_matrix(sub)

            sub_matrices_oe_big.append(oe_big)

    print("Matrix splitting completed.")

    # === Process each submatrix ===
    for idx, (sub_obs, sub_oe_big) in enumerate(zip(sub_matrices, sub_matrices_oe_big)):
        try:
            algo = get_stripe_algo_from_matrix(sub_obs, top_k, compare_eigenvalues)
            bkps = algo.predict(pen=penalty)[:-1]
        except Exception as e:
            print(f"Submatrix {idx} ruptures failed: {e}")
            continue

        global_bkps = [b + idx * step_size for b in bkps]
        all_changepoints.extend(global_bkps)

        if len(bkps) < 3:
            print(f"Submatrix {idx} has insufficient changepoints.")
            continue

        sub_oe = calculate_observed_expected_matrix(sub_obs)
        candidates = calculate_fold_changes(sub_oe, fold_thresh, max_width, min_length, bkps, merge_len_bins)
        filtered = intervals_limit(candidates, max_width, min_length)

        # === Tune stripe width ===
        tuned_stripes = []
        for stripe in filtered:
            width, length = stripe[:2], stripe[2]
            tuned_width, tuned_length = adjust_stripe_position(
                sub_oe_big, extend_bins, min_length, width, length,
                max_width, change_threshold=0.5
            )
            tuned_stripes.append((tuned_width, tuned_length))

        # === Perform statistical test ===
        for width, length in tuned_stripes:
            result = stripe_t_test_len_slant_within_out(
                sub_oe_big, width, length, extend_bins, len_select_bins
            )
            stripe_width, stripe_length = result[0], result[1]
            p_wid_up, p_wid_down, p_len = result[3:6]

            if None in (p_wid_up, p_wid_down, p_len):
                print(f"Stripe in submatrix {idx} skipped due to invalid p-values.")
                continue

            # Shift positions to global coordinates
            global_width = [stripe_width[0] + idx * step_size, stripe_width[1] + idx * step_size]
            global_length = [length + idx * step_size, stripe_length]

            stripe_results.append([
                global_width, global_length, result[2],  # width, length, direction
                p_wid_up, p_wid_down, p_len,             # p-values
                *result[6:9],                            # fold changes
                result[9], idx + 1                       # dip p-value, submatrix id
            ])

    return stripe_results, all_changepoints




def ad_pvalue(p_thresh_wid, p_thresh_len, raw_results, wid_up, wid_down, length):
    """
    Apply FDR correction to p-values and flag significant stripes.

    Parameters:
    - p_thresh_wid: float, adjusted p-value threshold for width (both up/down)
    - p_thresh_len: float, adjusted p-value threshold for length
    - raw_results: list of original result entries (each a list)
    - wid_up / wid_down / length: list-like of p-values to correct

    Returns:
    - revised_results: list of results with appended significance flags and adjusted p-values
    """
    wid_up = np.asarray(wid_up)
    wid_down = np.asarray(wid_down)
    length = np.asarray(length)

    if not (len(wid_up) and len(wid_down) and len(length)):
        print("One or more p-value arrays are empty. Skipping correction.")
        return []

    # FDR correction
    _, wid_up_adj = multipletests(wid_up, method='fdr_bh')[:2]
    _, wid_down_adj = multipletests(wid_down, method='fdr_bh')[:2]
    _, length_adj = multipletests(length, method='fdr_bh')[:2]

    revised_results = []
    for i, item in enumerate(raw_results):
        is_significant = int(
            wid_up_adj[i] <= p_thresh_wid and
            wid_down_adj[i] <= p_thresh_wid and
            length_adj[i] <= p_thresh_len
        )
        revised_results.append(item + [[is_significant, wid_up_adj[i], wid_down_adj[i], length_adj[i]]])

    return revised_results



def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input matrix not found: {path}")
    mat = np.loadtxt(path)
    # Guard against NaN/Inf
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def load_matrix_any(path: Path,
                    input_chrom: Optional[str] = None,
                    cool_norm: str = "weight") -> np.ndarray:
    """
    Load a matrix from either a text/mat file or a .cool file.

    - If path ends with .cool: require `input_chrom` (e.g. 'chr1' or 'chr1:10_000_000-20_000_000'),
      and fetch the balanced/unbalanced submatrix via Cooler.
    - Otherwise: treat as plain text matrix and np.loadtxt.
    """
    if path.suffix.lower() == ".cool":
        if cooler is None:
            raise ImportError("cooler is not installed but a .cool file was provided.")
        if not input_chrom:
            raise ValueError("When using a .cool file, `input_chrom` must be provided.")

        c = cooler.Cooler(str(path))

        # balance flag: 'weight' -> True; 'None' -> False; or a valid bins()
        if cool_norm == "None":
            balance_arg = False
        elif cool_norm == "weight":
            balance_arg = True
        else:
            if cool_norm in c.bins().columns:
                balance_arg = cool_norm
            else:
                raise ValueError(
                    f"`cool_norm` must be 'None', 'weight', or a bins() column in {path}. "
                    f"Available: {list(c.bins().columns)}"
                )

        mat = c.matrix(balance=balance_arg).fetch(input_chrom)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        return mat
    else:
        return load_matrix(path)
    

def call_stripes(
    mat: np.ndarray,
    penalty: float,
    fold_threshold1: float,
    split_length: int,
    step_size: int,
    max_width: int,
    min_length: int,
) -> Tuple[List[list], List[int]]:
    """Call stripes with or without matrix splitting, return (stripes, changepoints)."""
    top_k = 0
    if split_length and split_length > 0:
        stripes, bkps = split_and_call_stripes(
            mat, top_k, penalty, fold_threshold1, split_length, step_size, max_width, min_length
        )
    else:
        stripes, bkps = nosplit_and_call_stripes(
            mat, top_k, penalty, fold_threshold1, max_width, min_length
        )
    return stripes, bkps


def correct_pvalues(
    stripes: List[list], p_wid_cut: float, p_len_cut: float
) -> List[list]:
    """Correct p-values for stripes (in-place semantics per your engine)."""
    if not stripes:
        return stripes
    p_up = [x[3] for x in stripes]
    p_down = [x[4] for x in stripes]
    p_len = [x[5] for x in stripes]
    return ad_pvalue(p_wid_cut, p_len_cut, stripes, p_up, p_down, p_len)


def _safe_ge(x: float, thr: float) -> bool:
    try:
        return (x is not None) and np.isfinite(x) and (x >= thr)
    except Exception:
        return False


def apply_filters(
    stripes: List[list],
    fc_wid_cut: float,
    fc_len_cut: float,
    enable_dip: bool,
    dip_cut: float = 0.05,
) -> None:
    """
    Append two flags at the end of each stripe row:
      - pass_fc
      - pass_dip
    Assumes columns 6,7,8 are fc_wid_up, fc_wid_down, fc_len; column 9 is dip_p.
    """
    if not stripes:
        return

    for s in stripes:
        fc_ok = (_safe_ge(s[6], fc_wid_cut) and _safe_ge(s[7], fc_wid_cut) and _safe_ge(s[8], fc_len_cut))
        s.append(1 if fc_ok else 0)

    for s in stripes:
        dip_ok = (s[9] >= dip_cut) if enable_dip else True
        s.append(1 if dip_ok else 0)


def write_outputs(
    out_dir: Path,
    bkps: List[int],
    stripes: List[list],
    *,
    penalty: float,
    fold_threshold1: float,
    split_length: int,
    step_size: int,
    max_width: int,
    min_length: int,
    p_thresh_wid: float,
    p_thresh_len: float,
    fc_thresh_wid: float,
    fc_thresh_len: float,
    add_dip: str,
) -> Path:
    """
    Write changepoints + stripes table. Returns the stripes table path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # changepoints
    bkp_file = out_dir / f"changepoints_pen{penalty}_split{split_length}.csv"
    pd.DataFrame({"changepoints": bkps}).to_csv(bkp_file, index=False)

    # stripes table
    is_split = bool(split_length and split_length > 0)
    header = [
        "stripe_width",
        "stripe_len_anchor_and_length",
        "direction",
        "p_wid_up",
        "p_wid_down",
        "p_len",
        "fc_wid_up",
        "fc_wid_down",
        "fc_len",
        "dip_p",
    ]
    if is_split:
        header.append("split_mat_id")
    # Your engine already appends 'pass_t' earlier; here we only add pass_fc & pass_dip:
    header += ["pass_t", "pass_fc", "pass_dip"]

    result_file = out_dir / (
        "results_"
        f"pen{penalty}_fold{fold_threshold1}"
        f"_split{split_length}_step{step_size}"
        f"_wid{max_width}_len{min_length}"
        f"_p{p_thresh_wid}_{p_thresh_len}"
        f"_fc{fc_thresh_wid}_{fc_thresh_len}"
        f"_dip{add_dip.upper()}_unidentified_stripe.csv"
    )

    df = pd.DataFrame(stripes, columns=header) if stripes else pd.DataFrame(columns=header)
    df.to_csv(result_file, index=False)
    return result_file


def run_pipeline(
    input_matrix: Path,
    output_dir: Path,
    *,
    penalty: float = 0.1,
    fold_threshold1: float = 1.3,
    split_length: int = 200,
    step_size: int = 50,
    max_width: int = 8,
    min_length: int = 20,
    p_thresh_wid: float = 1e-3,
    p_thresh_len: float = 5e-2,
    fc_thresh_wid: float = 1.1,
    fc_thresh_len: float = 3.0,
    add_dip: str = "N",
    input_chrom: Optional[str] = None, 
    cool_norm: str = "weight",           # 'weight' | 'None' | bins()
) -> Path:
    """
    High-level function: load → call stripes → p-value correction → filters → write outputs.
    """
    if max_width <= 0 or min_length <= 0:
        raise ValueError("max_width and min_length must be positive.")
    if p_thresh_wid <= 0 or p_thresh_len <= 0:
        raise ValueError("p-value thresholds must be > 0.")
    if fc_thresh_wid <= 0 or fc_thresh_len <= 0:
        raise ValueError("fold-change thresholds must be > 0.")

    mat = load_matrix_any(input_matrix, input_chrom=input_chrom, cool_norm=cool_norm)

    stripes, bkps = call_stripes(
        mat,
        penalty=penalty,
        fold_threshold1=fold_threshold1,
        split_length=split_length,
        step_size=step_size,
        max_width=max_width,
        min_length=min_length,
    )

    stripes = correct_pvalues(stripes, p_thresh_wid, p_thresh_len)

    enable_dip = (add_dip.upper() == "Y")
    apply_filters(
        stripes,
        fc_wid_cut=fc_thresh_wid,
        fc_len_cut=fc_thresh_len,
        enable_dip=enable_dip,
        dip_cut=0.05,
    )

    return write_outputs(
        output_dir, bkps, stripes,
        penalty=penalty, fold_threshold1=fold_threshold1,
        split_length=split_length, step_size=step_size,
        max_width=max_width, min_length=min_length,
        p_thresh_wid=p_thresh_wid, p_thresh_len=p_thresh_len,
        fc_thresh_wid=fc_thresh_wid, fc_thresh_len=fc_thresh_len,
        add_dip=add_dip,
    )