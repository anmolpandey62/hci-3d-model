# config/classes.py

CLASS_NAMES = [
    "Room", "Window", "Door", "Wall", "Slab", "Roof", "Column", "Beam",
    "Stair", "Railing", "CurtainWall", "Furniture", "Covering",
    "LightFixture", "ElectricAppliance", "FlowTerminal", "EnergyConversionDevice"
]

CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}