NASA_CLASSES = {
    "snow_cover": range(1, 101),
    "no_snow": (0,),
    "clouds": (250,),
    "water": (237, 239),  # inland and ocean
    "no_decision": (201,),
    "night": (211,),
    "missing_data": (251,),
    "L1B_unusable": (252,),
    "bowtie_trim": (253,),
    "L1B_fill": (254,),
    "fill": (255,),
}

NODATA_NASA_CLASSES = (
    "no_decision",
    "night",
    "missing_data",
    "L1B_unusable",
    "bowtie_trim",
    "L1B_fill",
    "fill",
)

S2_CLASSES = {"snow_cover": range(1, 101), "no_snow": (0,), "clouds": (205,), "nodata": (255,), "fill": (255,)}
