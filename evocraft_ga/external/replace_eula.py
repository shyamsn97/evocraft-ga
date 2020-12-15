import click

@click.command()
@click.option("--eula_file_path", default="evocraft_ga/external/eula.txt", help="Path to eula.txt")
@click.option('--find_str', default="eula=false", help='string to replace')
@click.option('--replace_str', default="eula=true", help='value for replacement')
def edit_eula(eula_file_path, find_str, replace_str="eula=True"):
    with open(eula_file_path, 'r') as f:
        eula = f.read()

    print("Replacing {} with {}".format(find_str, replace_str))
    eula = eula.replace(find_str, replace_str)

    with open(eula_file_path, 'w') as f:
        f.writelines(eula)

if __name__ == '__main__':
    edit_eula()