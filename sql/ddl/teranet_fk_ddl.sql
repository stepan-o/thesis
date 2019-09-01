alter table teranet
	add constraint teranet_da_census_profiles_income_dauid_fk
		foreign key (dauid) references da_census_profiles_income;

alter table teranet
	add constraint teranet_da_census_select_tidy_dauid_year_fk
		foreign key (dauid, census2001_year) references da_census_select_tidy;

alter table teranet
	add constraint teranet_dmti_canfsa_fsa_fk
		foreign key (fsa) references dmti_canfsa;

alter table teranet
	add constraint teranet_dmti_postal_geography_pca_id_fk
		foreign key (pca_id) references dmti_postal_geography;

alter table teranet
	add constraint teranet_fuel_price_year_fk
		foreign key (tts_year) references fuel_price;

alter table teranet
	add constraint teranet_gen_da_dauid_fk
		foreign key (dauid) references gen_da (dauid);

alter table teranet
	add constraint teranet_gta_land_use_code_landuse_fk
		foreign key (landuse) references gta_land_use_code;

alter table teranet
	add constraint teranet_lu_gtha_pin_fk
		foreign key (pin_lu) references lu_gtha;

alter table teranet
	add constraint teranet_taz_info_taz_o_fk
		foreign key (taz_o) references taz_info;

alter table teranet
	add constraint teranet_tts_num_jobs_tidy_taz_id_year_fk
		foreign key (taz_o, tts1991_year) references tts_num_jobs_tidy;



