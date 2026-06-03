# user_data_optimisation

Usage
-----

Run the `flatten_pen_to_plane.py` script to flatten pen tracking data onto a
calibration plane. Typical commands:

- Process a single trial (pass the trial folder name / BORIS observation id):

	python flatten_pen_to_plane.py P003_Long_Large_Front_weighted_A180

- Process all trials for one or more participants (per-participant usage):

	python flatten_pen_to_plane.py --participants P003
	python flatten_pen_to_plane.py --participants P003,P004

- Preview which trials would be processed without running (dry-run):

	python flatten_pen_to_plane.py --participants P003 --dry-run

See the script's help for full options:

	python flatten_pen_to_plane.py --help

Label tracking
--------------

Use `03_label_tracking_with_boris.py` to label tracking CSVs with BORIS behaviours.

- Process a single trial folder:

	python 03_label_tracking_with_boris.py /path/to/trial_folder

- Process all trials under a landmarks root:

	python 03_label_tracking_with_boris.py /path/to/Participant_Landmarks --batch

- Process only specific participant(s):

	python 03_label_tracking_with_boris.py /path/to/Participant_Landmarks --participants P003
	python 03_label_tracking_with_boris.py /path/to/Participant_Landmarks --participants P003,P004

See `03_label_tracking_with_boris.py --help` for more options.