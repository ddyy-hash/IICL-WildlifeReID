#!/usr/bin/env python3

SPECIES_TYPE_MAP = {
    'atrw': 'stripe',
    'gzgc_zebra': 'stripe',
    'stripespotter': 'stripe',
    'nyala': 'stripe',

    'gzgc_giraffe': 'spot',
    'leopard': 'spot',

    'ipanda50': 'plain',
    'czechlynx': 'plain',
}


def get_species_type(dataset_name: str) -> str:
    """

    Args:

    Returns:
    """
    dataset_key = dataset_name.lower().replace('-', '_').replace(' ', '_')

    for key, species_type in SPECIES_TYPE_MAP.items():
        if key in dataset_key or dataset_key in key:
            return species_type

    print(f"Warning: Unknown dataset '{dataset_name}', defaulting to 'stripe'")
    return 'stripe'


def get_ipaid_params(dataset_name: str) -> dict:
    """

    Args:

    Returns:
    """
    species_type = get_species_type(dataset_name)

    params = {
        'base_channels': 32,
        'num_scales': 3,
        'refine_iterations': 2,
        'use_sensitivity': True,
        'use_refinement': True,
        'use_feature_guided': True,
        'species_type': species_type,
    }

    if species_type == 'spot':
        params['use_color_illumination'] = True
        print(f"[Config] Dataset '{dataset_name}' is spot-type: Color illumination will be disabled")
    elif species_type == 'stripe':
        params['use_color_illumination'] = True
    else:  # plain
        params['use_color_illumination'] = False

    return params


if __name__ == '__main__':
    print("Species Type Configuration")
    print("=" * 60)

    for dataset_name in SPECIES_TYPE_MAP.keys():
        species_type = get_species_type(dataset_name)
        params = get_ipaid_params(dataset_name)

        print(f"\n{dataset_name}:")
        print(f"  Species Type: {species_type}")
        print(f"  Color Illumination: {params['use_color_illumination']}")
        print(f"  Feature Guided: {params['use_feature_guided']}")
