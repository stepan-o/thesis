select 
	t.transaction_id as transaction_id,
	t.pin as t_pin,
	lu.pin as lu_pin,
	t.unitno as unit,
	t.street_number as st_num,
	t.street_name as st_name,
	luc.code as landuse,
	lu.aream2 as LUpin_area_m2,
	t.registration_date as rdate,
	t.price_2016 as price2016,
	pg.pca_count as pca_count,
	cs.avg_hhinc16 as da_hhinc16,
	nj."2016" as tts_numjobs_2016,
	tazi.shape_area as taz_shape_area,
	fp.fuel_price_per_litre as fuel_price_per_litre
from teranet as t
join lu_gtha as lu
	on t.pin_lu=lu.pin
join dmti_postal_geography as pg
	on t.pca_id=pg.pca_id
join da_census_select as cs
	on t.dauid=cs.dauid
join taz_info as tazi
	on t.taz_o=tazi.taz_o
join tts_num_jobs as nj
	on t.taz_o=nj.taz_id
join gta_land_use_code as luc
	on lu.landuse=luc.landuse
join fuel_price as fp
	on t."year"=fp."year"
where t."year"=2016
order by t.registration_date
limit 100;