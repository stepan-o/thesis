-- auto-generated definition
create table lu_gtha
(
    aream2     numeric,
    depth      numeric,
    distance   numeric,
    fid_1      integer,
    fid_2      numeric,
    frontage   numeric,
    id         numeric,
    landuse    integer,
    lengthm    numeric,
    objectid   integer,
    pin        integer not null
        constraint lu_gtha_pk
            primary key,
    pin_1      integer,
    prop_code  integer,
    shape_area numeric,
    shape_leng numeric,
    site_area  numeric,
    municipali text
);

comment on table lu_gtha is 'Parcel-level Land Use from GTA and Hamilton';

alter table lu_gtha
    owner to teranet;

