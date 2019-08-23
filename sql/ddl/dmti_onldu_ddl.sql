create table dmti_onldu_nodup
(
	pca_id integer not null
		constraint onldu_nodup_pk
			primary key,
	postalcode char(6),
	prov char(2),
	maf_id integer,
	prec_code numeric,
	pca_count numeric,
	dom_pca numeric,
	multi_pc numeric,
	del_m_id text,
	longitude numeric,
	latitude numeric,
	geometry text
);

comment on table dmti_onldu_nodup is 'DMTI: Postal Code geography';

alter table dmti_onldu_nodup owner to teranet;