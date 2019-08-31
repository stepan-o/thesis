ALTER TABLE da_census_select_tidy
    ADD PRIMARY KEY (dauid, year);

ALTER TABLE tts_num_jobs_tidy
    ADD PRIMARY KEY (taz_id, year);
