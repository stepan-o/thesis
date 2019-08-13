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
    objectid                  integer,
    dauid                     integer
        constraint teranet_da_census_profiles_income_dauid_fk
            references da_census_profiles_income,
    csdname                   varchar(22),
    street_name_raw           varchar(50),
    date_disp                 date,
    price_disp                text,
    year                      integer,
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
    xy_sale_next_3y           boolean
);

comment on table teranet is 'Teranet dataset filtered for GTHA records with price over 10''000 CAD';

alter table teranet
    owner to teranet;

