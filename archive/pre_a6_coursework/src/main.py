def main(category_points: list[float]) -> float:
    """Return the total points from all assignment categories."""
    return sum(category_points)


if __name__ == "__main__":  # pragma: no cover
    main([])
