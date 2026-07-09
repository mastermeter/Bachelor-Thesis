# Bachelor Thesis

This repository containt all the code related to my bachelor thesis.

The main subject is the cross-view geolocalization apply to swiss datas.

## Swiss dataset

One objective was to produce a dataset adapted to cross-view geolocalization (street-view and satellite). Because images are not really adapted to be committed to Github here is a swisstransfer link : [Swisstransfer link](https://www.swisstransfer.com/d/cc758d1d-44d4-4fcf-910f-a62399f987ad)

To select all the images you will find in the dataset, I selected specific areas in mapilary where the density of 360-view were relevant. Then, I manually select a border on a [website](https://geojson.io/) to generate a file that containt the area selected. If you want to add new specific area. You will have to provide a new file with new coordinates and adapt 'GEOJSON_INPUT' with the name of the new file you provide. You can find, as example, 3 files which contain an area of 3 places in Switzerland(Lausanne, Zurich, Aigle).
