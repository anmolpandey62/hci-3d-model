# logic/ifc_properties.py
# IFC Property Sets (Psets) schema for floor plan elements.
# Covers: Rooms, Doors, Windows, Furniture, Sanitary, Appliances.

IFC_SCHEMA = {
    "Room": {
        "ifc_class": "IfcSpace",
        # IFC4: IfcSpace.PredefinedType
        "predefined_types": ["SPACE", "PARKING", "GFA", "INTERNAL", "EXTERNAL"],
        # Subtypes = IfcSpace.LongName mapped to Uniclass 2015 SL codes
        "subtypes": [
            "Living Room (SL_25_10_30)",
            "Dining Room (SL_25_10_47)",
            "Drawing Room (SL_25_10_30)",
            "Bedroom (SL_25_15_10)",
            "Master Bedroom (SL_25_15_30)",
            "Kids Bedroom (SL_25_15_10)",
            "Guest Bedroom (SL_25_15_10)",
            "Kitchen (SL_25_30_30)",
            "Kitchenette (SL_25_30_30)",
            "Bathroom (SL_25_20_10)",
            "Toilet / WC (SL_25_20_74)",
            "Powder Room (SL_25_20_74)",
            "Balcony (SL_25_60_10)",
            "Terrace (SL_25_60_47)",
            "Entry / Foyer (SL_25_35_47)",
            "Corridor (SL_25_35_30)",
            "Lobby (SL_25_35_47)",
            "Utility / Wash (SL_25_40_74)",
            "Store Room (SL_25_40_74)",
            "Study / Home Office (SL_25_10_74)",
            "Puja Room (SL_25_10_74)",
            "Parking / Garage (SL_25_50_47)",
            "Servant Room (SL_25_15_10)",
            "Gym (SL_25_10_30)",
        ],
        "psets": {
            "Pset_SpaceCommon": {
                "OccupancyType":  {"type": "select", "options": [
                    "LIVING", "DINING", "BEDROOM", "KITCHEN", "BATHROOM",
                    "TOILET", "BALCONY", "CORRIDOR", "LOBBY", "UTILITY",
                    "STORE", "STUDY", "PARKING", "PUJA", "GYM", "OTHER"
                ]},
                "FloorFinish":    {"type": "select", "options": [
                    "Marble", "Vitrified Tiles", "Anti-skid Tiles",
                    "Wooden Flooring", "Granite", "Ceramic Tiles",
                    "Epoxy", "Carpet", "Kota Stone", "IPS"
                ]},
                "WallFinish":     {"type": "select", "options": [
                    "Plastic Emulsion", "Texture Paint", "Tiles",
                    "Wallpaper", "Exposed Brick", "Lime Plaster", "Distemper"
                ]},
                "CeilingFinish":  {"type": "select", "options": [
                    "POP", "Gypsum Board", "PVC Panel", "Wooden",
                    "Exposed Concrete", "False Ceiling"
                ]},
                "IsExternal":     {"type": "bool",   "default": False},
                "GrossFloorArea": {"type": "number", "unit": "m2"},
                "NetFloorArea":   {"type": "number", "unit": "m2"},
                "Height":         {"type": "number", "unit": "m", "default": 2.7},
                "FireRating":     {"type": "select", "options": ["None", "30 min", "60 min", "120 min"]},
            },
            "Pset_SpaceThermal": {
                "VentilationType":    {"type": "select", "options": ["Natural", "Mechanical", "Mixed", "None"]},
                "AirChangesPerHour":  {"type": "number"},
                "DesignTemperature":  {"type": "number", "unit": "C"},
            },
            "Pset_SpaceLighting": {
                "LightingLevel":      {"type": "number", "unit": "lux"},
                "ArtificialLighting": {"type": "bool"},
            },
            "Pset_SpaceAcoustic": {
                "AcousticRating":  {"type": "select", "options": ["Low", "Medium", "High"]},
                "SoundInsulation": {"type": "number", "unit": "dB"},
            },
        }
    },
    "Door": {
        "ifc_class": "IfcDoor",
        "subtypes": ["Single Swing", "Double Swing", "Sliding", "Folding", "Revolving", "Flush"],
        "psets": {
            "Pset_DoorCommon": {
                "OperationType":  {"type": "select", "options": ["SINGLE_SWING_LEFT", "SINGLE_SWING_RIGHT", "DOUBLE_SWING", "SLIDING", "FOLDING", "REVOLVING", "FIXED"]},
                "OverallWidth":   {"type": "number", "unit": "m", "default": 0.9},
                "OverallHeight":  {"type": "number", "unit": "m", "default": 2.1},
                "Material":       {"type": "text",   "options": ["Teak Wood", "Flush Door", "UPVC", "Aluminium", "Glass", "Steel"]},
                "Finish":         {"type": "text",   "options": ["Painted", "Polished", "Laminated", "Veneer", "Anodized"]},
                "FireRating":     {"type": "text",   "options": ["None", "30 min", "60 min", "90 min", "120 min"]},
                "IsExternal":     {"type": "bool",   "default": False},
                "SecurityRating": {"type": "text",   "options": ["Low", "Medium", "High"]},
            }
        }
    },
    "Window": {
        "ifc_class": "IfcWindow",
        "subtypes": ["Sliding", "Casement", "Awning", "Fixed", "Louvered", "Bay"],
        "psets": {
            "Pset_WindowCommon": {
                "OperationType":  {"type": "select", "options": ["SLIDING", "CASEMENT", "AWNING", "FIXED", "LOUVERED", "TILT_AND_TURN"]},
                "OverallWidth":   {"type": "number", "unit": "m", "default": 1.2},
                "OverallHeight":  {"type": "number", "unit": "m", "default": 1.2},
                "Material":       {"type": "text",   "options": ["Aluminium", "UPVC", "Wood", "Steel"]},
                "GlazingType":    {"type": "text",   "options": ["Single", "Double", "Triple", "Tinted", "Frosted"]},
                "ThermalTransmittance": {"type": "number", "unit": "W/m²K"},
                "IsExternal":     {"type": "bool",   "default": True},
            }
        }
    },
    "Wall": {
        "ifc_class": "IfcWall",
        "subtypes": ["Exterior", "Interior", "Partition", "Retaining", "Curtain"],
        "psets": {
            "Pset_WallCommon": {
                "Material":       {"type": "text",   "options": ["Brick", "AAC Block", "RCC", "Drywall", "Glass"]},
                "Thickness":      {"type": "number", "unit": "mm", "default": 230},
                "FireRating":     {"type": "text",   "options": ["None", "30 min", "60 min", "120 min"]},
                "IsExternal":     {"type": "bool",   "default": False},
                "LoadBearing":    {"type": "bool",   "default": True},
                "Finish":         {"type": "text",   "options": ["Plaster", "Tiles", "Paint", "Exposed Brick"]},
            }
        }
    },
    "Furniture": {
        "ifc_class": "IfcFurnishingElement",
        "subtypes": ["Sofa", "Bed", "Wardrobe", "Dining Table", "Chair", "TV Unit",
                     "Study Table", "Bookshelf", "Cabinet", "Coffee Table"],
        "psets": {
            "Pset_FurnitureTypeCommon": {
                "Style":          {"type": "text",   "options": ["Modern", "Contemporary", "Classic", "Minimalist", "Industrial"]},
                "Material":       {"type": "text",   "options": ["Teak Wood", "MDF", "Plywood", "Metal", "Rattan", "Acrylic"]},
                "Finish":         {"type": "text",   "options": ["Laminate", "Veneer", "Paint", "Polish", "Fabric", "Leather"]},
                "Color":          {"type": "color",  "default": "#8B4513"},
                "OverallWidth":   {"type": "number", "unit": "m"},
                "OverallDepth":   {"type": "number", "unit": "m"},
                "OverallHeight":  {"type": "number", "unit": "m"},
                "Manufacturer":   {"type": "text"},
                "ModelNumber":    {"type": "text"},
            },
            "Pset_SofaTypeCommon": {
                "SeatingCapacity":    {"type": "number"},
                "UpholsteryMaterial": {"type": "text", "options": ["Leather", "Fabric", "Velvet", "Microfiber"]},
                "FrameMaterial":      {"type": "text", "options": ["Solid Wood", "Metal", "Plywood"]},
            },
            "Pset_BedTypeCommon": {
                "BedSize":    {"type": "select", "options": ["Single", "Double", "Queen", "King", "Super King"]},
                "StorageType":{"type": "text",   "options": ["None", "Hydraulic", "Drawer", "Box"]},
            },
        }
    },
    "FlowTerminal": {
        "ifc_class": "IfcSanitaryTerminal",
        "subtypes": ["WC / Toilet", "Wash Basin", "Kitchen Sink", "Shower", "Bathtub", "Urinal"],
        "psets": {
            "Pset_SanitaryTerminalTypeCommon": {
                "SanitaryTerminalType": {"type": "select", "options": ["TOILETPAN", "WASHHANDBASIN", "SINK", "SHOWER", "BATH", "URINAL"]},
                "Material":    {"type": "text",   "options": ["Ceramic", "Porcelain", "Acrylic", "Stainless Steel", "Cast Iron"]},
                "Color":       {"type": "color",  "default": "#FFFFFF"},
                "Mounting":    {"type": "select", "options": ["Wall-hung", "Floor-mounted", "Counter-top", "Under-mount"]},
                "FlushType":   {"type": "text",   "options": ["Single Flush", "Dual Flush", "Sensor"]},
                "WaterResistance": {"type": "text", "options": ["Low", "Medium", "High"]},
                "Manufacturer":{"type": "text",   "options": ["Kohler", "Jaquar", "Hindware", "Cera", "American Standard"]},
            }
        }
    },
    "ElectricAppliance": {
        "ifc_class": "IfcElectricAppliance",
        "subtypes": ["Refrigerator", "Washing Machine", "Dishwasher", "Microwave",
                     "Stove / Hob", "AC Unit", "Water Heater", "Exhaust Fan"],
        "psets": {
            "Pset_ElectricApplianceTypeCommon": {
                "ApplianceType": {"type": "select", "options": ["REFRIGERATOR", "WASHINGMACHINE", "DISHWASHER", "MICROWAVE", "STOVE", "AIRCONDITIONER", "WATERHEATER", "FAN"]},
                "PowerRating":   {"type": "number", "unit": "W"},
                "Voltage":       {"type": "number", "unit": "V", "default": 230},
                "Color":         {"type": "color",  "default": "#C0C0C0"},
                "Manufacturer":  {"type": "text",   "options": ["Samsung", "LG", "Whirlpool", "Bosch", "IFB", "Voltas", "Daikin"]},
                "ModelNumber":   {"type": "text"},
                "EnergyRating":  {"type": "select", "options": ["1 Star", "2 Star", "3 Star", "4 Star", "5 Star"]},
                "Capacity":      {"type": "text",   "unit": "L or kg"},
                "FuelType":      {"type": "select", "options": ["Electric", "Gas", "LPG", "Solar"]},
            }
        }
    },
    "Stair": {
        "ifc_class": "IfcStair",
        "subtypes": ["Straight", "L-shaped", "U-shaped", "Spiral", "Winder"],
        "psets": {
            "Pset_StairCommon": {
                "NumberOfRiser":  {"type": "number"},
                "NumberOfTreads": {"type": "number"},
                "RiserHeight":    {"type": "number", "unit": "mm", "default": 175},
                "TreadLength":    {"type": "number", "unit": "mm", "default": 250},
                "Material":       {"type": "text",   "options": ["RCC", "Steel", "Wood", "Marble", "Granite"]},
                "FireRating":     {"type": "text",   "options": ["None", "30 min", "60 min", "120 min"]},
            }
        }
    },
    "Column": {
        "ifc_class": "IfcColumn",
        "subtypes": ["RCC", "Steel", "Timber", "Composite"],
        "psets": {
            "Pset_ColumnCommon": {
                "Material":   {"type": "text",   "options": ["RCC", "Steel", "Timber", "Composite"]},
                "Width":      {"type": "number", "unit": "mm"},
                "Depth":      {"type": "number", "unit": "mm"},
                "FireRating": {"type": "text",   "options": ["None", "30 min", "60 min", "120 min"]},
            }
        }
    },
    "Slab": {
        "ifc_class": "IfcSlab",
        "subtypes": ["Floor Slab", "Roof Slab", "Stair Landing", "Ramp"],
        "psets": {
            "Pset_SlabCommon": {
                "Material":   {"type": "text",   "options": ["RCC", "Precast", "Composite"]},
                "Thickness":  {"type": "number", "unit": "mm", "default": 150},
                "FireRating": {"type": "text",   "options": ["None", "30 min", "60 min", "120 min"]},
                "Finish":     {"type": "text",   "options": ["Smooth", "Rough", "Polished"]},
            }
        }
    },
    "LightFixture": {
        "ifc_class": "IfcLightFixture",
        "subtypes": ["Ceiling Light", "Pendant", "Recessed", "Wall Sconce", "Floor Lamp", "Track Light"],
        "psets": {
            "Pset_LightFixtureTypeCommon": {
                "LightFixtureType": {"type": "select", "options": ["POINTSOURCE", "DIRECTIONSOURCE", "SECURITYLIGHTING", "EMERGENCYLIGHTING"]},
                "Wattage":    {"type": "number", "unit": "W"},
                "LampType":   {"type": "select", "options": ["LED", "CFL", "Halogen", "Fluorescent", "Incandescent"]},
                "Color":      {"type": "color",  "default": "#FFFFE0"},
                "ColorTemp":  {"type": "select", "options": ["Warm White (2700K)", "Neutral White (4000K)", "Cool White (6500K)"]},
                "Lumens":     {"type": "number", "unit": "lm"},
                "Manufacturer": {"type": "text"},
            }
        }
    },
}

# Material library for IfcMaterial
MATERIALS = {
    "Concrete":        {"color": [0.7, 0.7, 0.7], "category": "Structure"},
    "Brick":           {"color": [0.8, 0.4, 0.2], "category": "Masonry"},
    "Marble":          {"color": [0.95, 0.95, 0.95], "category": "Finish"},
    "Teak Wood":       {"color": [0.55, 0.27, 0.07], "category": "Wood"},
    "Ceramic":         {"color": [0.9, 0.9, 0.9], "category": "Finish"},
    "Glass":           {"color": [0.7, 0.9, 1.0], "category": "Transparent"},
    "Steel":           {"color": [0.75, 0.75, 0.8], "category": "Metal"},
    "Aluminium":       {"color": [0.85, 0.85, 0.85], "category": "Metal"},
    "Granite":         {"color": [0.4, 0.4, 0.4], "category": "Finish"},
    "Plywood":         {"color": [0.82, 0.71, 0.55], "category": "Wood"},
    "Stainless Steel": {"color": [0.8, 0.8, 0.82], "category": "Metal"},
    "Porcelain":       {"color": [0.95, 0.95, 0.95], "category": "Finish"},
    "Acrylic":         {"color": [0.9, 0.95, 1.0], "category": "Plastic"},
    "Leather":         {"color": [0.4, 0.2, 0.1], "category": "Fabric"},
    "Fabric":          {"color": [0.6, 0.6, 0.8], "category": "Fabric"},
}


def get_schema(cls_name: str) -> dict:
    return IFC_SCHEMA.get(cls_name, {})


def get_default_pset(cls_name: str) -> dict:
    """Return default property values for a class."""
    schema = get_schema(cls_name)
    result = {}
    for pset_name, props in schema.get("psets", {}).items():
        result[pset_name] = {}
        for prop, meta in props.items():
            if "default" in meta:
                result[pset_name][prop] = meta["default"]
    return result


def validate_pset(cls_name: str, pset_data: dict) -> dict:
    """Validate and clean pset data against schema."""
    schema = get_schema(cls_name)
    cleaned = {}
    for pset_name, props in pset_data.items():
        if pset_name not in schema.get("psets", {}):
            continue
        cleaned[pset_name] = {}
        for prop, value in props.items():
            if prop in schema["psets"][pset_name]:
                cleaned[pset_name][prop] = value
    return cleaned
