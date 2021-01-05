

# from dbfread import DBF

# records = []
# for record in DBF('/home/shreekanthajith/intello_satellite/ForAfric-Agricultural-Fields-Delineation-master/data/marker2016_small.dbf'):
# 	records.append(record)
	
# for i in range(10):
# 	print (records[i])

import geopandas as gpd
shapefile = gpd.read_file("/home/shreekanthajith/intello_satellite/ForAfric-Agricultural-Fields-Delineation-master/data/marker2016_small.shp")
print(shapefile)
