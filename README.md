# Bachelor Thesis

This repository containt all the code related to my bachelor thesis.

The main subject is the cross-view geolocalization (CVGL) apply to swiss datas.

## ConGeo

For this work and after a week of research about this subject, I decided to work on ConGeo. A model built by a collaboration of Wuhan University and EPFL. This repository directly forked their work.

- [Their paper](https://arxiv.org/pdf/2403.13965)
- [Their repository](https://github.com/eceo-epfl/ConGeo)

Their main objective was to provide an efficient tool that can be able to deal with images that are not totally adapted to cross-view geolocalization. More precisely, they manage to have strong result with images that have a non-standard configuration such as reduced FoV (field of view).

## Usage

Because of the large dataset I used and the process required to train a model, I got access to the infrastructure of my school called Dance. To be able to efficiently use this ressource, I had to setup an apptainer and launch my program by slurm jobs. This is why you will find a .def file (that can be considered an equivalent to a dockerfile) and many .sh.

To build the image based on the .def file you will need to write this first :

```bash
apptainer build <name_of_your_image>.sif rules.def
```

It will take a couple of minutes to finish this process.

After that, you can now run slurm job (which are the .sh file) with this command :

```bash
sbatch --export=ALL,USER_NAME=<your_session_name_in_Dance> <name_of_sh_file>.sh
```
 
 What I consider "your session name" is the name of the folder after home/ in Dance. The name of the sh file in our case would be either : `job_vigor.sh`, `job_sw.sh`, `job_topk.sh`

 ### Placement of the datasets

 To make training available, you will need to place the datasets I used (VIGOR, a well-known dataset for CVGL and a datatset of my own creation focused on Swizerland) and they both need to be placed in the folder `datasets`. Of course, they also need to be unzip if you get them as zip archive (or tar.gz). Be carefull for VIGOR that subfolder may also be zipped as I recieved them like that too. 

## Swiss dataset

One objective was to produce a dataset adapted to cross-view geolocalization (street-view and satellite). Because images are not really adapted to be committed to Github here is a swisstransfer link : [Swisstransfer link](https://www.swisstransfer.com/d/cc758d1d-44d4-4fcf-910f-a62399f987ad)

To select all the images you will find in the dataset, I selected specific areas in Mapillary where the density of 360-view were relevant. Then, I manually select a border on a [website](https://geojson.io/) to generate a file that containt the area selected. If you want to add new specific area. You will have to provide a new file with new coordinates and adapt 'GEOJSON_INPUT' with the name of the new file you provide. You can find, as example, 3 files which contain an area of 3 places in Switzerland(Lausanne, Zurich, Aigle).

You will also need to create an account to Mapillary and add an app in your [developper workspace](https://www.mapillary.com/developer?locale=fr_FR) to get a token.

For theses programs specifically, I didn't use Dance but download them more classicaly. If you want to use it you will have to create a virtual environment and add the libraires listed in the `requirements.txt` file.

## Application on non standard view.

As a conclusion to this work. I had the possibility to use RTS archive to gather images to test the efficiency of the system. For that, I also created a small dataset with some images that could be considered "street-view" and get the corresponding satellite view the same way that with my swiss datatset. I also get satellite images around the objective to see the efficiency of the system. A archive of this small experimental dataset can also be found in swisstransfer : [Link](https://www.swisstransfer.com/d/458acd10-c194-4795-97c3-0238a8f83b5f)
