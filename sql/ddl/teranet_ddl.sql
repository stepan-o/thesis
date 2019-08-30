-- auto-generated definition
create table teranet
(
    transaction_id            integer not null
        constraint teranet_pk
            primary key,
    lro_num                   integer,
    pin                       integer,
    consideration_amt         numeric,
    registration_date         date,
    postal_code               char(6),
    unitno                    varchar(30),
    street_name               varchar(50),
    street_designation        varchar(7),
    street_direction          char,
    municipality              varchar(22),
    street_number             integer,
    x                         numeric,
    y                         numeric,
    dauid                     integer
        constraint teranet_da_census_profiles_income_dauid_fk
            references da_census_profiles_income
        constraint teranet_da_census_select_dauid_fk
            references da_census_select,
    csduid                    integer,
    csdname                   varchar(22),
    taz_o                     integer
        constraint teranet_taz_info_taz_o_fk
            references taz_info
        constraint teranet_tts_num_jobs_taz_id_fk
            references tts_num_jobs,
    fsa                       char(3)
        constraint teranet_dmti_canfsa_fsa_fk
            references dmti_canfsa,
    pca_id                    integer
        constraint teranet_dmti_onldu_nodup_pca_id_fk
            references dmti_postal_geography,
    postal_code_dmti          char(6),
    maf_id                    integer,
    del_m_id                  char(6),
    pin_lu                    integer
        constraint teranet_lu_gtha_pin_fk
            references lu_gtha,
    landuse                   integer,
    prop_code                 integer,
    street_name_raw           varchar(50),
    date_disp                 date,
    price_disp                varchar(17),
    year                      integer
        constraint teranet_temporal_spans_year_fk
            references temporal_spans,
    year_month                char(7),
    year3                     varchar(9),
    year5                     varchar(9),
    year10                    varchar(9),
    xy                        varchar(38),
    pin_total_sales           integer,
    xy_total_sales            integer,
    pin_prev_sales            integer,
    xy_prev_sales             integer,
    pin_price_cum_sum         numeric,
    xy_price_cum_sum          numeric,
    pin_price_pct_change      numeric,
    xy_price_pct_change       numeric,
    price_da_pct_change       numeric,
    pin_years_since_last_sale numeric,
    xy_years_since_last_sale  numeric,
    da_days_since_last_sale   numeric,
    da_years_since_last_sale  numeric,
    pin_sale_next_6m          boolean,
    pin_sale_next_1y          boolean,
    pin_sale_next_3y          boolean,
    xy_sale_next_6m           boolean,
    xy_sale_next_1y           boolean,
    xy_sale_next_3y           boolean,
    price_2016                numeric
);

comment on table teranet is 'Teranet records, with new foreign keys, cleaned and filtered for GTHA and price >10''''000CAD';

alter table teranet
    owner to teranet;

