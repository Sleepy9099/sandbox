from fs.mount import mount

with open("vendor.bin", "rb") as f:
    fs, info = mount(f, offset=0x0, fs_type="auto")
    print(info)

    print([e.name for e in fs.ls("/")])
    st = fs.stat("build.prop")
    print(st)

    with fs.open("build.prop") as fh:   # if you add __enter__/__exit__ later
        print(fh.read(200))


with open("exfat_test.img", "rb") as f:
    fs, info = mount(f, offset=0x0, fs_type="auto")
    print(info)

    print([e.name for e in fs.ls("/dirA")])
    st = fs.stat("hello.txt")
    print(st)

    with fs.open("hello.txt") as fh:   # if you add __enter__/__exit__ later
        print(fh.read(200))