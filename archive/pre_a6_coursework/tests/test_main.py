from src.main import main


def test_main():
    assert main([1, 2, 3, 4, 5]) == 15
    assert main([]) == 0
