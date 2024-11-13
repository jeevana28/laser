# new File intervention_script.py:
import os

# Function to run the intervention command with specified parameters
def run_intervention(lname, lnum, rate):
    command = f'python src/intervention_blip_coco.py --lname "{lname}" --rate {rate} --lnum {lnum}'
    os.system(command)

# if __name__ == "__main__":
#     # Run the script
#     # Iterate over lnums from 1 to 11
#     for lnum in range(1, 12):
#         # Iterate over rates 8, 8.5, 9, and 9.5
#         for rate in [0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 9.9]:
#             for lname in ["fc_out", "fc_in", "out_proj", "v_proj", "q_proj", "k_proj"]:
#                 print(f"Running intervention with lnum={lnum} and rate={rate}")
#                 run_intervention(lname, lnum, rate)

##only for mlp layers

if __name__ == "__main__":
    # Run the script
    # Iterate over lnums from 1 to 11
    for lnum in range(1, 12):
        # Iterate over rates 8, 8.5, 9, and 9.5
        for rate in [9, 8, 7, 6, 5, 4, 3, 2, 1]:
            for lname in ["fc_out", "fc_in"]:
                print(f"Running intervention with lnum={lnum} and rate={rate}")
                run_intervention(lname, lnum, rate)