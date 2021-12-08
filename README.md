
# Statistical model of the COVID-19 vaccination campaign

(NOTE: an **interactive web App** version of this model can be found [here](https://covid19-vaccination-app.davidfcastellanos.com))

This **statistical model** reproduces the characteristics of the evolution of the ongoing COVID-19 **vaccination campaign** in specific countries. With it, the user can test the effects of different vaccine production and **delivery rates**, the segmentation of the population into **pro- and anti-vaccines**, and the impact of **social pressure** on non-vaccinated people. Moreover, the model can handle uncertainty in the parameters, allowing the user to consider **worst and best-case scenarios**.

This is an example of the model results, in this case optimized to reproduce the vaccination campaign in USA between the beginning of the campaign till the end of 2021:

![image](https://user-images.githubusercontent.com/5737365/144921956-da822382-1631-4c40-b409-0c762a2d66f5.jpg)

<p align="center"><i>
The model results are shown in blue, where the shaded region contains 95% of the sampled data. The results are optimized to reproduce the USA vaccination campaign (in red). The real-world data is obtained from Our World in Data.
</i></p>

## Motivation
Vaccination campaigns can be regarded as relatively simple to model from a statistical perspective. One reason is that vaccination campaigns typically proceed as designed by professionals, away from the media, and supported by a great deal of experience accumulated from years of repetition. In these conditions, **sociological factors** are expected to play a minor role. However, these conditions do not entirely hold in the case of COVID-19 for several reasons:

-   Society is **fully aware** of the process, arguably drawing the biggest mediatic attention.    
-   Society is **fragmented** into groups with different views on vaccines, trying to influence each other's opinions.    
-   The vaccines are produced, delivered, and applied almost in real-time, so the vaccination dynamics must be coupled with **production and delivery dynamics**.    
-   The **uncertainty in the conditions** under which the vaccination is going to take place in a fastly evolving scenario of competition between countries and new virus variants
    
These points are the main reasons that make the COVID-19 vaccination campaign special from a statistical modeling perspective.  

## The model
The model and the sampling procedure are explained in detail in `assets/model_explanation.html`.


## Command-line interface

The commands are:

	  -h, --help            show this help message and exit
	  --pro PRO             comma-separated upper and lower bounds for the probability that a certain person belongs to the pro-vaccines group
	  --anti ANTI           comma-separated upper and lower bounds for the probability that a specific person belongs to the anti-vaccines group
	  --pressure PRESSURE   comma-separated upper and lower bounds for the strength of the social pressure effect
	  --dupl_time DUPL_TIME
	                        comma-separated upper and lower bounds for the duplication time of the weekly arriving vaccines
	  --init_stock INIT_STOCK
	                        comma-separated upper and lower bounds for the initial stock of vaccines, measured as a percentage of the population size
	  --max_delivery MAX_DELIVERY
	                        comma-separated upper and lower bounds for the maximum weekly delivery capacity, measured as a percentage over the population size
	  --mc_samples MC_SAMPLES
	                        number of Monte Carlo samples (optional)
	  --date_range DATE_RANGE
	                        comma-separated starting and ending dates
	  --CI CI               value of the quantile used for establishing the confidence intervals (optional)

This example call reproduces the results shown in the picture above:
	
	python model.py --pro=35,45 --anti=12,25 --pressure=0.,0.02 --dupl_time=5,7 --init_stock=1,1.2 --max_delivery=5,10 --date_range=2020-12-15,2022-07-1

## Documentation
The source files are fully documented with docstrings.

## Related links

-   The author's website: [https://www.davidfcastellanos.com](https://www.davidfcastellanos.com)
-   An interactive web App version: [https://covid19-vaccination-app.davidfcastellanos.com](https://covid19-vaccination-app.davidfcastellanos.com)
-   An associated blog post with extra information: [https://www.davidfcastellanos.com/covid-19-vaccination-model](https://www.davidfcastellanos.com/covid-19-vaccination-model)
-   Real world data: [https://github.com/owid/covid-19-data](https://github.com/owid/covid-19-data)

## License
This software is open source. You can freely use it, redistribute it, and/or modify it
under the terms of the Creative Commons Attribution 4.0 International Public 
License. The full text of the license can be found in the file LICENSE at the top level of the distribution.
 
Copyright (C) 2021  - David Fernández Castellanos.

