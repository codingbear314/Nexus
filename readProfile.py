import pstats

# Load and print the profile stats
stats = pstats.Stats('output_file.out')
stats.strip_dirs()  # Optional: Clean up long file paths
stats.sort_stats('time')  # Sort by total time spent
stats.print_stats(10)  # Print the top 10 results