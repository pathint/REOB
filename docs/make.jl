push!(LOAD_PATH,"../src/")

using Documenter
using REOBiomarker

const DOCS_ASSETS = [
    Documenter.RawHTMLHeadContent(
        """
        <style>
        .math-display {
            margin: 1rem 0;
            text-align: center;
        }
        .math-display math {
            font-size: 1.08rem;
        }
        math {
            font-family: "Latin Modern Math", "STIX Two Math", "Cambria Math", serif;
        }
        </style>
        """
    ),
    Documenter.RawHTMLHeadContent(
        """
        <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
        """
    ),
    "assets/docs-ui.js",
    "assets/docs-mermaid.js",
    Documenter.RawHTMLHeadContent(
        """
        <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23eff1f5'/%3E%3Cpath d='M14 35c9-14 27 14 36 0' fill='none' stroke='%2304a5e5' stroke-width='6' stroke-linecap='round'/%3E%3Cpath d='M14 26c9 14 27-14 36 0' fill='none' stroke='%23883ef1' stroke-width='6' stroke-linecap='round'/%3E%3C/svg%3E">
        """
    ),
]

makedocs(
    sitename = "REOBiomarker Docs",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = nothing,
        repolink = nothing,
        inventory_version = "0.1.0",
        assets = DOCS_ASSETS,
    ),
    modules = [REOBiomarker],
    remotes = nothing,
    pages = [
        "Overview" => "index.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

# Documenter expects siteinfo.js and versions.js for versioned deployments.
# This project uses single-version local preview; writing lightweight
# placeholders avoids 404 noise from static file servers.
const BUILD_DIR = joinpath(@__DIR__, "build")
write(
    joinpath(BUILD_DIR, "siteinfo.js"),
    """
    window.DOCUMENTER_CURRENT_VERSION = "v0.1.0";
    window.DOCUMENTER_STABLE = "v0.1.0";
    window.DOCUMENTER_IS_DEV_VERSION = false;
    window.DOCUMENTER_VERSION_SELECTOR_DISABLED = true;
    """,
)
write(
    joinpath(BUILD_DIR, "versions.js"),
    """
    window.DOCUMENTER_NEWEST = "v0.1.0";
    window.DOCUMENTER_STABLE = "v0.1.0";
    window.DOCUMENTER_VERSIONS = ["v0.1.0"];
    """,
)

function write_le16(io::IO, value::Integer)
    write(io, UInt8(value & 0xff))
    write(io, UInt8((value >> 8) & 0xff))
end

function write_le32(io::IO, value::Integer)
    for shift in (0, 8, 16, 24)
        write(io, UInt8((value >> shift) & 0xff))
    end
end

function write_favicon_ico(path::AbstractString)
    width = 16
    height = 16
    bitmap = IOBuffer()

    # ICO files use a BMP payload with a DIB header; pixels are BGRA, bottom-to-top.
    write_le32(bitmap, 40)
    write_le32(bitmap, width)
    write_le32(bitmap, 2 * height)
    write_le16(bitmap, 1)
    write_le16(bitmap, 32)
    write_le32(bitmap, 0)
    write_le32(bitmap, width * height * 4)
    write_le32(bitmap, 0)
    write_le32(bitmap, 0)
    write_le32(bitmap, 0)
    write_le32(bitmap, 0)

    for y in (height - 1):-1:0
        for x in 0:(width - 1)
            r, g, b = if abs(x - y) <= 1
                (0x88, 0x39, 0xef)
            elseif abs(x + y - (width - 1)) <= 1
                (0x04, 0xa5, 0xe5)
            else
                (0xef, 0xf1, 0xf5)
            end

            write(bitmap, UInt8(b))
            write(bitmap, UInt8(g))
            write(bitmap, UInt8(r))
            write(bitmap, UInt8(0xff))
        end
    end

    write(bitmap, zeros(UInt8, 4 * height))
    image = take!(bitmap)

    icon = IOBuffer()
    write_le16(icon, 0)
    write_le16(icon, 1)
    write_le16(icon, 1)
    write(icon, UInt8(width))
    write(icon, UInt8(height))
    write(icon, UInt8(0))
    write(icon, UInt8(0))
    write_le16(icon, 1)
    write_le16(icon, 32)
    write_le32(icon, length(image))
    write_le32(icon, 22)
    write(icon, image)

    write(path, take!(icon))
end

write_favicon_ico(joinpath(BUILD_DIR, "favicon.ico"))
