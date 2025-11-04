import pandera as pa
from pandera import Column, Check

CO2Schema = pa.DataFrameSchema({
    "ts": Column(pa.Timestamp, coerce=True, nullable=False),
    "zone": Column(str, checks=Check.isin(["DK1", "DK2"])),
    "co2_g_per_kwh": Column(float, checks=Check.ge(0)),
})
