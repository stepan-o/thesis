create table gen_da as
	select 
		cs.dauid as dauid,
		round(avg(t.price_2016), 2) as avg_price_corr,
		round(avg(case when t.price_2016 > 5000000 then null else t.price_2016 end), 2) as avg_price16_limit_corr,
		percentile_cont(0.5) within group (order by t.price_2016)::numeric as median_price_corr,
		count(t.price_2016) as teranet_count,
		cs.avg_rent16 as avg_rent16,  
		cs.avg_hhinc16 as avg_hhinc16, 
		cs.municipality as municipality
	from da_census_select as cs
	left join teranet as t -- left join will keep DAs that have no Teranet records in 2016
		on cs.dauid=t.dauid
	    and t.year=2016
	group by cs.dauid
	order by dauid;
		
alter table gen_da
add constraint gen_da_da_census_select_dauid_fk foreign key (dauid) references da_census_select(dauid);

alter table gen_da
add column median_tot_inc16 numeric;

update gen_da
set median_tot_inc16 = ci.median_tot_inc
from da_census_profiles_income as ci
where gen_da.dauid=ci.dauid;

alter table gen_da
add column avg_da_days_since_last_sale numeric;

update gen_da
set avg_da_days_since_last_sale = tagg.avg_da_days_since_last_sale
from (
	select 
		dauid as dauid,
		avg(da_days_since_last_sale) as avg_da_days_since_last_sale
	from teranet
	where year=2016
	group by dauid
) as tagg
where gen_da.dauid=tagg.dauid;