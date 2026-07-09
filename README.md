# Bachelor Thesis

This repository containt all the code related to my bachelor thesis.

The main subject is the cross-view geolocalization apply to swiss datas.

## Swiss dataset

One objective was to produce a dataset adapted to cross-view geolocalization (street-view and satellite). Because images are not really adapted to be committed to Github here is a swisstransfer link : [Swisstransfer link](https://www.swisstransfer.com/d/cc758d1d-44d4-4fcf-910f-a62399f987ad)

To select all the images you will find in the dataset, I selected specific areas in Mapillary where the density of 360-view were relevant. Then, I manually select a border on a [website](https://geojson.io/) to generate a file that containt the area selected. If you want to add new specific area. You will have to provide a new file with new coordinates and adapt 'GEOJSON_INPUT' with the name of the new file you provide. You can find, as example, 3 files which contain an area of 3 places in Switzerland(Lausanne, Zurich, Aigle).

You will also need to create an account to Mapillary and add an app in your [developper workspace](https://www.mapillary.com/developer?locale=fr_FR) to get a token.


## Application on non standard view.

As a conclusion to this work. I had the possibility to use RTS archive to gather images to test the efficiency of the system. For that, I also created a small dataset with some images that could be considered "street-view" and get the corresponding satellite view the same way that with my swiss datatset. I also get satellite images around the objective to see the efficiency of the system. A archive of this small experimental dataset can also be found in swisstransfer : [Link](https://www.swisstransfer.com/d/458acd10-c194-4795-97c3-0238a8f83b5f)
