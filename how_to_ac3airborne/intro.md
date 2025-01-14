# Introduction
Welcome to the world of AC3 airborne data! It's a collection of python code examples to get you started with the airborne data. It is a permantently growing and changing document and might never be finished. It follows the concept and ideas of the work done within the [EUREC4A](https://howto.eurec4a.eu/intro.html) community.

## Idea
The book chapters show datasets that are accessible online, i.e., you don’t have to download anything in advance. Most datasets are accessed via the [ac3airborne intake catalog](https://github.com/igmk/ac3airborne-intake), which simply said takes care of the links to datasets in their most recent version. By implemented caching capabilities, it is only necessary to download the data the first time they are used. The scripts typically contain at minimum how to get a specific dataset and some simple plots of basic quantities. Most chapters include additional information from aircraft flight segments or further meta data, sometimes more sophisticated plots, or also a combination of variables from different datasets. In addition, some small tools are included to work with auxilliary data like sea ice coverage or land-mask. 

## Airborne campaigns within AC3

The campaigns handled in the online description are:

- Arctic CLoud Observations Using airborne measurements during polar Day [ACLOUD](https://home.uni-leipzig.de/~ehrlich/ACLOUD_wiki_doku/doku.php) (data at {cite:t}`EhrlichCollectionDataSources2019`) 
- Airborne measurements of radiative and turbulent FLUXes of energy and momentum in the Arctic boundary layer [AFLUX](https://home.uni-leipzig.de/~ehrlich/AFLUX_wiki_doku/doku.php?id=start) (data at {cite:t}`MechCollectionDataSets2021`)
- MOSAiC Airborne observations in the Central Arctic [MOSAiC-ACA](https://home.uni-leipzig.de/~ehrlich/MOSAiC_ACA_wiki_doku/doku.php?id=start) (data at {cite:t}`MechCollectionDataSets2021a`)

```{figure} img/tracks.png
---
height: 250px
name: directive-fig
---
Flight tracks of the ACLOUD (left), AFLUX (center), and MOSAiC-ACA (right) campaigns.
```

During [ACLOUD](https://home.uni-leipzig.de/~ehrlich/ACLOUD_wiki_doku/doku.php) the campaign Physical feedback of Arctic PBL, Sea ice, Cloud And AerosoL (PASCAL) was conducted by the research vessel Polarstern. This will be important for analyzing Polarstern overflights.

## Some links
* [AC3 webpage](http://www.ac3-tr.de/)
* The data description papers of ACLOUD from {cite:t}`EhrlichComprehensiveSituRemote2019` and for AFLUX and MOSAiC-ACA from {cite:t}`MechMOSAiCACAAFLUXArctic2021`
* Campaign wikis: [ACLOUD](https://home.uni-leipzig.de/~ehrlich/ACLOUD_wiki_doku/doku.php), [AFLUX](https://home.uni-leipzig.de/~ehrlich/AFLUX_wiki_doku/doku.php?id=start), and [MOSAiC-ACA](https://home.uni-leipzig.de/~ehrlich/MOSAiC_ACA_wiki_doku/doku.php?id=start)
* Master data collections on PANGAEA: ACLOUD from {cite:t}`EhrlichCollectionDataSources2019`, AFLUX from {cite:t}`MechCollectionDataSets2021`, and MOSAiC-ACA from {cite:t}`MechCollectionDataSets2021a`