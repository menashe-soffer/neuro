import os
import random
import sys

# It's good practice to import libraries only if they are needed.
# This ensures that you don't have to install tqdm on the server.
try:
    from tqdm import tqdm

    IS_ON_SERVER = False
except ImportError:
    # This block will be executed if tqdm is not installed, which is
    # a good way to tell if we are on the server or on a machine
    # without this library.
    IS_ON_SERVER = True
    print("tqdm not found. Assuming script is running on the server.")

from multiprocessing import Pool


def process_folder(folder_path, param1, param2):
    """
    Processes a single folder with additional parameters.

    Args:
        folder_path (str): The path to the folder to process.
        param1 (str): A constant parameter for all iterations.
        param2 (int): Another constant parameter for all iterations.

    Returns:
        dict: A dictionary containing collected statistics for this folder.
    """
    # Your core processing logic for this folder goes here.
    # We'll use a print statement and some dummy statistics for this example.
    if IS_ON_SERVER:
        print(f"Processing folder: {os.path.basename(folder_path)} "
              f"with param1='{param1}' and param2={param2}")

    # Simulate some work
    # time.sleep(random.uniform(0.1, 1.0))

    # Collect statistics for this folder and return them
    stats = {
        'folder': os.path.basename(folder_path),
        'files_processed': random.randint(10, 50),
        'result_metric': random.random()
    }

    return stats


if __name__ == '__main__':
    # 1. Define your parameters (the same for all iterations)
    my_param1 = "value_A"
    my_param2 = 123

    # 2. Prepare the list of tasks (one tuple per folder)
    base_data_path = '/path/to/my_data/'
    # In a real scenario, you'd use os.listdir to get your list of folders
    # For this example, we'll create a dummy list
    folders_to_process = [f'folder_{i}' for i in range(10)]

    # Create a list of tuples: [(path, param1, param2), (path, param1, param2), ...]
    tasks = [(os.path.join(base_data_path, folder), my_param1, my_param2) for folder in folders_to_process]

    # 3. Use conditional logic to run with a progress bar on the PC
    # and with multiprocessing on the server.
    if not IS_ON_SERVER:
        # PC: Use a simple loop with a progress bar
        print("Running on PC: Using a simple loop with progress bar.")
        all_stats = []
        # Wrap the list of tasks with tqdm to get a progress bar
        for task in tqdm(tasks, desc="Processing folders"):
            stats = process_folder(*task)
            all_stats.append(stats)
    else:
        # Server: Use multiprocessing
        print("Running on server: Using multiprocessing pool.")
        # Set the number of cores to match your #BSUB -n flag
        num_cores = os.cpu_count() or 4
        print(f"Starting parallel processing on {num_cores} cores...")
        with Pool(num_cores) as p:
            all_stats = p.starmap(process_folder, tasks)

    # 4. Collect and process the results
    # The 'all_stats' list now contains the dictionaries returned by each worker
    total_files_processed = sum(s['files_processed'] for s in all_stats)
    print("\n------------------------------------")
    print("All tasks complete.")
    print(f"Total files processed: {total_files_processed}")
    print("------------------------------------")

    # You can now save or analyze the collected stats from the 'all_stats' list,
    # for example, saving them to a CSV file or a database.
    # import pandas as pd
    # df = pd.DataFrame(all_stats)
    # df.to_csv('analysis_summary.csv')
