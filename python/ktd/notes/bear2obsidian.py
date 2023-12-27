#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import shutil
from datetime import datetime
from urllib.parse import unquote

import regex as re
import yaml


# NOTE: src_dir is the location of the bear export; new_asset_dir should
# correspond to the 'Files and links' > 'Default location for
# attachments' in Obsidian.
#
# NOTE: It also makes most sense to set 'New link format' to 'Relative
# path to file' under the same 'Files and links' section.
def process(
    src_dir: str,
    tgt_dir: str,
    new_asset_dir: str = "assets",
    min_modified_date: datetime = datetime.min,
):
    os.makedirs(f"{tgt_dir}/{new_asset_dir}", exist_ok=True)

    link_pattern = re.compile(r"\[([^][]*)\](\(((?:[^()]+|(?2))+)\))")
    wikilink_pattern = re.compile(r"\[([^][]+)\]")

    def should_include(fn):
        modified_date = datetime.fromtimestamp(os.stat(f"{src_dir}/{fn}").st_mtime)
        return osp.splitext(fn)[-1] == ".md" and (modified_date > min_modified_date)

    md_fns = [p for p in os.listdir(src_dir) if should_include(p)]
    print(
        f"{len(md_fns)} files are newer than the minimum modification date of {min_modified_date}"
    )
    for md_fn in md_fns:
        print(f"Processing file '{md_fn}'")
        # paths
        src_md_path = f"{src_dir}/{md_fn}"
        tgt_md_path = f"{tgt_dir}/{md_fn}"

        # create output path
        os.makedirs(f"{tgt_dir}/{new_asset_dir}", exist_ok=True)

        # read source markdown
        with open(src_md_path, "r") as f:
            md = f.read()

        # process all markdown links
        for _, _, src_rel_path_quoted in link_pattern.findall(md):
            src_rel_path = unquote(src_rel_path_quoted)
            src_path = f"{src_dir}/{src_rel_path}"

            # ignore if it's not a link to a local file
            if not osp.exists(src_path):
                continue

            # copy file, preserving stats, to the target asset directory
            # and update links to this file
            tgt_rel_path = (
                f"{new_asset_dir}/{src_rel_path.replace('/', '~').replace(' ', '_')}"
            )
            tgt_path = f"{tgt_dir}/{tgt_rel_path}"
            shutil.copyfile(src_path, tgt_path)
            shutil.copystat(src_path, tgt_path)

            md = md.replace(src_rel_path_quoted, tgt_rel_path)

        # add import-/export-related tags as yaml frontmatter
        stat = os.stat(src_md_path)
        # NOTE: this is platform-specific; confirmed on MacOS
        creation_date = datetime.fromtimestamp(stat.st_birthtime)
        modified_date = datetime.fromtimestamp(stat.st_mtime)
        front_matter = {
            "creation_date": f"{creation_date:%Y-%m-%d-%H-%M-%S}",
            "modified_date": f"{modified_date:%Y-%m-%d-%H-%M-%S}",
            "tags": ["bear-import"],
        }
        md = f"---\n{yaml.dump(front_matter)}---\n{md}"

        # write modified markdown file, preserving stats
        with open(tgt_md_path, "w") as f:
            f.write(md)
        shutil.copystat(src_md_path, tgt_md_path)

        # add any linked MD files we haven't seen to the list for processing
        linked_md_fns = [f"{fn}.md" for fn in wikilink_pattern.findall(md)]
        for linked_md_fn in linked_md_fns:  # slow; fine
            if linked_md_fn not in md_fns and osp.exists(f"{src_dir}/{linked_md_fn}"):
                print(f"\tAdding file '{linked_md_fn}', linked from '{md_fn}'")
                md_fns.append(linked_md_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument(
        "min_modified_date",
        type=str,
        default=datetime.min.strftime("%Y-%m-%d"),
        help="In YYYY-MM-DD format",
    )

    args = parser.parse_args()
    process(
        src_dir=args.source,
        tgt_dir=args.target,
        min_modified_date=datetime.fromisoformat(args.min_modified_date),
    )


if __name__ == "__main__":
    main()
