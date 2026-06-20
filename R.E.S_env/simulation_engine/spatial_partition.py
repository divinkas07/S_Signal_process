"""
Spatial Partitioning – Grid-based Earth cell indexing for fast interactions.
Optimizes visibility checks by only testing satellites against local and neighboring cells.
"""

import numpy as np


class SpatialGrid:
    """
    Divides the Earth surface into a grid of cells (latitude/longitude bins).
    Used to quickly find which ground stations are near a satellite.
    """

    def __init__(self, cell_size_deg: float = 10.0):
        self.cell_size = cell_size_deg
        # 360 / 10 = 36 cols, 180 / 10 = 18 rows
        self.n_lat = int(180 / cell_size_deg)
        self.n_lon = int(360 / cell_size_deg)
        
        # Grid format: grid[(lat_idx, lon_idx)] = [station_id1, station_id2, ...]
        self.grid = {}
        # Reverse lookup: obj_id -> (lat_idx, lon_idx)
        self.objects = {}

    def _get_cell_idx(self, lat_deg: float, lon_deg: float) -> tuple:
        """Convert lat/lon to grid indices."""
        # Latitude from -90 to +90 -> 0 to n_lat-1
        lat_idx = min(int((lat_deg + 90) / self.cell_size), self.n_lat - 1)
        lat_idx = max(lat_idx, 0)
        
        # Longitude from -180 to +180 -> 0 to n_lon-1
        lon_idx = min(int((lon_deg + 180) / self.cell_size), self.n_lon - 1)
        lon_idx = max(lon_idx, 0)
        
        return lat_idx, lon_idx

    def add_object(self, obj_id: str, lat_deg: float, lon_deg: float):
        """Add or update an object's position in the grid."""
        cell_idx = self._get_cell_idx(lat_deg, lon_deg)
        
        if obj_id in self.objects:
            old_cell = self.objects[obj_id]
            if old_cell == cell_idx:
                return  # Hasn't moved cells
            self.grid[old_cell].remove(obj_id)
            
        self.objects[obj_id] = cell_idx
        
        if cell_idx not in self.grid:
            self.grid[cell_idx] = set()
        self.grid[cell_idx].add(obj_id)

    def remove_object(self, obj_id: str):
        """Remove an object from the grid."""
        if obj_id in self.objects:
            cell_idx = self.objects[obj_id]
            self.grid[cell_idx].remove(obj_id)
            del self.objects[obj_id]

    def get_nearby_objects(self, lat_deg: float, lon_deg: float, radius_deg: float) -> list:
        """
        Get all objects within a bounding box centered on the given coordinates.
        This provides a fast pre-filter before strict distance checking.
        """
        center_lat_idx, center_lon_idx = self._get_cell_idx(lat_deg, lon_deg)
        cell_radius = int(np.ceil(radius_deg / self.cell_size))
        
        nearby = []
        
        for dlat in range(-cell_radius, cell_radius + 1):
            lat_idx = center_lat_idx + dlat
            if lat_idx < 0 or lat_idx >= self.n_lat:
                continue
                
            for dlon in range(-cell_radius, cell_radius + 1):
                lon_idx = (center_lon_idx + dlon) % self.n_lon  # wrap around longitude
                
                cell = (lat_idx, lon_idx)
                if cell in self.grid:
                    nearby.extend(self.grid[cell])
                    
        return nearby

    def clear(self):
        """Clear the spatial grid."""
        self.grid.clear()
        self.objects.clear()
