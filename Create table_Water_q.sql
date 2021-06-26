create table `water_quality`.`water_q`(
`pH` decimal(18,2) not null,
`Hardness` decimal(18,2) not null,
`Solids` decimal(18,2) not null,
`Chloramines` decimal(18,2) not null,
`Sulfate` decimal(18,2) not null,
`Conductivity` decimal(18,2) not null,
`Organic_carbon` decimal(18,2) not null,
`Trihalomethanes` decimal(18,2) not null,
`Turbidity` decimal(18,2) not null,
`Potability` int not null
);