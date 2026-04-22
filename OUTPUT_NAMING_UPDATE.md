# Reframed output naming update

This version preserves the calculation logic from the original pipeline and updates
human-facing output names to match the revised manuscript framing.

Main changes:
- Renamed output Excel files and Supplementary_Data_S1 sheet names.
- Replaced legacy framing terms such as "evolution", "synergy", and "leaf_structure"
  in output-facing table names with descriptive manuscript terms.
- Renamed exported column headers for readability:
  - p_obs -> observed_p_severe
  - p_exp -> expected_p_severe
  - Bliss_mean -> mean_delta_bliss
  - n_obs -> observations_used
  - 1/2 -> B_round1/B_round2
  - B -> barrier_summary
- Updated README and command-line description.
- No equations, filters, groupings, model specifications, or statistical calculations
  were changed.
