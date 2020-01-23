# Clearsky Dictionary Creation
## Introduction
This respository contains data and code used to construct a "clearsky dictionary". A full description of the algorithm and data can be found [here](overleaf.com/1577517535shnfkqjdbrcm). Files are sorted by type into `code` and `data`.

## `code`
The `code` directory contains all code used in the project. Code is further broken up by `matlab`, `python`, and `bash`. See the `python` subdirectory for further description of the code contained there, as well as instructions for compiling the CSD. Scripts in `bash` are to be used in conjunction with some scripts in `python`, and so description of the bash scripts can also be found in `python`.

## `data`
The `data` directory contains all data used in the project. Several data sources can be found here, including the raw TSIs (found in `TSI_images`) as well as pyranometer measurements provided by NREL's SRRL group (found in `clearsky_index`). The `TSI_images` directory contains not only the structured image files (contained in `jpegs` and ordered so that the path to each TSI indicates the date on which it was taken, as well as the camera which took it), but also a zip file containing all TSIs, an unstuctured directory containing the originally-named copies of the TSIs (`cloud_images`), and some python scripts which be used to unpack the TSIs from `cloud_images` (found in `utils`).

If you are accessing this code through GitHub then the image files will not be included, as they are too large to push through Git. Instead the `utils` directory is included, and a zip file of the images can be made available upon request. A description of how to do this can be found in the `utils` README file.

