### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 48d3db35-9f3d-4496-b61f-fff070454067
using JSON

# ╔═╡ 4f71ec7d-cfb3-4efa-b2c9-3066328c81fb
using PlutoUI

# ╔═╡ f5d44736-f052-11eb-1bb6-4b71ff25a6d7
datadir = "/home/paethon/git/ADAPET/data/fewglue"

# ╔═╡ 3e06a548-2ed0-4e27-a3f5-138c9d4b3a8b
configdir = "/home/paethon/git/ADAPET/config"

# ╔═╡ afb7d78f-0540-4037-a581-87802de8faef
@bind dataset Select(["BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC", "WSC"])

# ╔═╡ 399c9a24-02fb-4d36-8dda-f1eb5c3bd151
begin
	jsl = JSON.parse.(readlines(joinpath(datadir,dataset,"train.jsonl")))
	jsl[1]
end

# ╔═╡ 6b6fae31-eb49-4119-ad48-5c7ee659c9e4
begin
	jsunl = JSON.parse.(readlines(joinpath(datadir, dataset, "unlabeled.jsonl")))
	jsunl[1]
end

# ╔═╡ e7f8f05e-17c0-44a9-b8ca-263e3b729884
jsconf = JSON.parse(read(joinpath(configdir, dataset)*".json", String))

# ╔═╡ c217c7ba-911e-4791-96e1-0b202b84f76c


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
JSON = "~0.21.1"
PlutoUI = "~0.7.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
"""

# ╔═╡ Cell order:
# ╠═48d3db35-9f3d-4496-b61f-fff070454067
# ╠═4f71ec7d-cfb3-4efa-b2c9-3066328c81fb
# ╠═f5d44736-f052-11eb-1bb6-4b71ff25a6d7
# ╠═3e06a548-2ed0-4e27-a3f5-138c9d4b3a8b
# ╠═afb7d78f-0540-4037-a581-87802de8faef
# ╠═399c9a24-02fb-4d36-8dda-f1eb5c3bd151
# ╠═6b6fae31-eb49-4119-ad48-5c7ee659c9e4
# ╠═e7f8f05e-17c0-44a9-b8ca-263e3b729884
# ╠═c217c7ba-911e-4791-96e1-0b202b84f76c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
