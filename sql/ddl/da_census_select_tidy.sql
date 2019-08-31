-- auto-generated definition
create table da_census_select_tidy
(
    dauid        integer not null,
    year         integer not null,
    avg_hhsize   numeric,
    avg_hhinc    numeric,
    avg_own_payt numeric,
    avg_val_dwel numeric,
    avg_rent     numeric,
    pop          integer,
    popdens      numeric,
    dwel         numeric,
    dweldens     numeric,
    sgl_det      numeric,
    apt_5plus    numeric,
    sgl_att      numeric,
    owned        numeric,
    rented       numeric,
    cartrvan_d   numeric,
    cartrvan_p   numeric,
    pt           numeric,
    walk         numeric,
    bike         numeric,
    lbrfrc       numeric,
    emp          numeric,
    unemp        numeric,
    not_lbrfrc   numeric,
    employee     numeric,
    self_emp     numeric,
    at_home      numeric,
    no_fix_wkpl  numeric,
    usl_wkpl     numeric,
    blue_cljob   numeric,
    white_cljob  numeric,
    constraint da_census_select_tidy_pkey
        primary key (dauid, year)
);

comment on table da_census_select_tidy is 'Select Census variables in a tidy format (with year as a variable)';

alter table da_census_select_tidy
    owner to teranet;

-- composite primary key can be added with:
--ALTER TABLE da_census_select_tidy
--    ADD PRIMARY KEY (dauid, year);
